use clap::Parser;
use forward_forward::Mat;
use mnist::{IMAGE_HEIGHT, IMAGE_WIDTH, Mnist, NPIXELS, error::MnistError};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// --- Hyperparameters ---
const USE_DROPOUT: bool = true;
const USE_AUGMENTATION: bool = true;
const SANITISE: bool = false;
const TINY: f32 = 1e-10;
const NUMLAB: usize = 10;
const TEMP: f32 = 1.0;
const LABELSTRENGTH: f32 = 1.0;
const MINLEVELSUP: usize = 2;
const WC: f32 = 0.002;
const SUPWC: f32 = 0.003;
const EPSILON: f32 = 0.01;
const EPSILONSUP: f32 = 0.1;
const DELAY: f32 = 0.9;
const MAX_EPOCH: usize = 200;
const LAYERS: [usize; 4] = [784, 1000, 1000, 1000];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Config {
    #[arg(short, long="dir", default_value_t = String::from("MNIST"))]
    /// X-axis range: min,max
    dir: String,

    #[arg(long, default_value_t = 0.15)]
    pub dropout: f32,

    #[arg(long, default_value_t = 0.03)]
    pub lambda_mean: f32,

    //// Using Vec allows flexible network depth, unlike [usize; 4]
    //#[arg(long, value_delimiter = ',', default_value = "784,1000,1000,1000")]
    //pub layers: Vec<usize>,
    #[arg(long, default_value_t = 100)]
    pub batch_size: usize,
}

// --- Data Structures ---

struct Layer {
    // --- Model Parameters ---
    weights: Mat,
    biases: Vec<f32>,
    supweights: Option<Mat>,

    // --- Optimizer State (Momentum/Velocity) ---
    // These store the running average of gradients to smooth updates
    weight_velocity: Mat,
    biases_grad: Vec<f32>,
    sup_weight_velocity: Option<Mat>,

    // --- Normalization State ---
    // Stores the running average of neuron activity (activations)
    // Used to punish neurons that are always on or always off
    activity_running_mean: Vec<f32>,
}

pub struct FFModel {
    layers: Vec<Layer>,
}

impl FFModel {
    // We only need Config here to create the initial random weights
    pub fn new(rng: &mut SmallRng) -> Self {
        let layers: Vec<Layer> = (0..LAYERS.len() - 1)
            .map(|i| {
                let fanin = LAYERS[i];
                let fanout = LAYERS[i + 1];
                Layer {
                    weights: Mat::new_randn(fanin, fanout, 1.0 / (fanin as f32).sqrt(), rng),
                    biases: vec![0.0; fanout],
                    supweights: Some(Mat::zeros(fanout, NUMLAB)),
                    weight_velocity: Mat::zeros(fanin, fanout),
                    biases_grad: vec![0.0; fanout],
                    sup_weight_velocity: Some(Mat::zeros(fanout, NUMLAB)),
                    activity_running_mean: vec![0.5; fanout],
                }
            })
            .collect();
        FFModel { layers }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        // Just save the number of layers.
        // The matrices themselves self-describe their dimensions.
        writer.write_all(&(self.layers.len() as u64).to_le_bytes())?;

        for layer in &self.layers {
            layer.weights.write_raw(&mut writer)?;

            writer.write_all(&(layer.biases.len() as u64).to_le_bytes())?;
            // endianness risk...
            let b_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    layer.biases.as_ptr() as *const u8,
                    layer.biases.len() * 4,
                )
            };
            writer.write_all(b_bytes)?;

            if let Some(sw) = &layer.supweights {
                writer.write_all(&[1u8])?;
                sw.write_raw(&mut writer)?;
            } else {
                writer.write_all(&[0u8])?;
            }
        }
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<FFModel> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let num_layers = u64::from_le_bytes(buf) as usize;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Mat::read_raw reads the rows/cols, so we recover topology automatically
            let weights = Mat::read_raw(&mut reader)?;

            reader.read_exact(&mut buf)?;
            let b_len = u64::from_le_bytes(buf) as usize;
            let mut biases = vec![0.0f32; b_len];
            let b_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u8, b_len * 4)
            };
            reader.read_exact(b_bytes)?;

            let mut opt = [0u8; 1];
            reader.read_exact(&mut opt)?;
            let supweights = if opt[0] == 1 {
                Some(Mat::read_raw(&mut reader)?)
            } else {
                None
            };

            // Reconstruct the training buffers based on the loaded weights
            let (rows, cols) = (weights.rows, weights.cols);
            layers.push(Layer {
                weight_velocity: Mat::zeros(rows, cols),
                biases_grad: vec![0.0; cols],
                sup_weight_velocity: supweights.as_ref().map(|sw| Mat::zeros(sw.rows, sw.cols)),
                activity_running_mean: vec![0.5; cols],
                weights,
                biases,
                supweights,
            });
        }
        Ok(FFModel { layers })
    }
}

struct BatchWorkspace {
    data: Mat,
    targets: Mat,
    lab_data: Mat,
    labin: Mat,
    dc_din_sup: Mat,
    neg_data: Mat,
    pos_st: Vec<Mat>,
    pos_nst: Vec<Mat>,
    neg_st: Vec<Mat>,
    neg_nst: Vec<Mat>,
    softmax_nst: Vec<Mat>,
    pos_probs: Vec<Vec<f32>>,
    pos_dc_din: Vec<Mat>,
    neg_dc_din: Vec<Mat>,
    pos_dw: Vec<Mat>,
    neg_dw: Vec<Mat>,
    sup_contrib: Mat,
    softmax_st: Vec<Mat>,
    sw_grad_tmp: Mat,
}

impl BatchWorkspace {
    fn new(layers: &[usize], batch_size: usize) -> Self {
        // Shape: [Batch Size x Layer Width] (for all layers 0..N)
        let nst_template: Vec<Mat> = layers.iter().map(|&c| Mat::zeros(batch_size, c)).collect();
        let st_template = nst_template[1..].to_vec(); // for hidden layers only
        let dw_template: Vec<Mat> = (0..layers.len() - 1)
            .map(|i| Mat::zeros(layers[i], layers[i + 1]))
            .collect();

        BatchWorkspace {
            data: Mat::zeros(batch_size, layers[0]),
            targets: Mat::zeros(batch_size, NUMLAB),
            lab_data: Mat::zeros(batch_size, layers[0]),
            labin: Mat::zeros(batch_size, NUMLAB),
            dc_din_sup: Mat::zeros(batch_size, NUMLAB),
            neg_data: Mat::zeros(batch_size, layers[0]),

            pos_st: st_template.clone(),
            neg_st: st_template.clone(),
            pos_dc_din: st_template.clone(),
            neg_dc_din: st_template.clone(),
            softmax_st: st_template,

            pos_nst: nst_template.clone(),
            neg_nst: nst_template.clone(),
            softmax_nst: nst_template.clone(),

            pos_probs: vec![vec![0.0; batch_size]; layers.len() - 1],
            pos_dw: dw_template.clone(),
            neg_dw: dw_template,
            sup_contrib: Mat::zeros(batch_size, NUMLAB),
            sw_grad_tmp: Mat::zeros(*layers.iter().max().unwrap(), NUMLAB),
        }
    }
}

// --- Helper Functions ---

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn goodness(row: &[f32], temp: f32) -> f32 {
    let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
    let cols = row.len() as f32;
    sigmoid((sum_sq - cols) / temp)
}

fn update_batch_goodness(st_mat: &Mat, temp: f32, probs_out: &mut [f32]) {
    let cols = st_mat.cols;
    let d = cols as f32;

    st_mat
        .data
        .par_chunks_exact(cols)
        .zip(probs_out.par_iter_mut())
        .for_each(|(row, p)| {
            let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
            *p = sigmoid((sum_sq - d) / temp);
        });
}

#[inline(always)]
fn sanitise_slice(data: &mut [f32]) {
    if SANITISE {
        for x in data.iter_mut() {
            if x.is_nan() {
                *x = 0.0;
            }
            *x = x.clamp(-1e10, 1e10);
        }
    }
}

/// Applies MatMul -> Bias -> ReLU -> Dropout -> Normalization
fn layer_io_into(
    cfg: &Config,
    vin: &Mat,
    layer: &Layer,
    st: &mut Mat,
    nst: &mut Mat,
    orng: Option<&mut SmallRng>,
) {
    vin.matmul_into(&layer.weights, st);
    let cols = st.cols;

    // Process Bias + ReLU + Dropout
    if let Some(rng) = orng
        && USE_DROPOUT
    {
        let dropout_scale: f32 = 1.0 / (1.0f32 - cfg.dropout);
        st.data.chunks_exact_mut(cols).for_each(|row| {
            for (val, &bias) in row.iter_mut().zip(layer.biases.iter()) {
                *val = (*val + bias).max(0.0); // ReLU
                *val = if rng.random::<f32>() < cfg.dropout {
                    0.0
                } else {
                    *val * dropout_scale
                };
            }
        });
    } else {
        // Inference path (Parallelizable)
        st.data.par_chunks_mut(cols).for_each(|row| {
            for (val, &bias) in row.iter_mut().zip(layer.biases.iter()) {
                *val = (*val + bias).max(0.0);
            }
        });
    }
    nst.data.copy_from_slice(&st.data);
    nst.norm_rows();
}

const MAX_SHIFT: i32 = 1;

fn apply_random_shift(src_image: &[f32; NPIXELS], target_buffer: &mut [f32], rng: &mut SmallRng) {
    let shift_x = rng.random_range(-MAX_SHIFT..=MAX_SHIFT);
    let shift_y = rng.random_range(-MAX_SHIFT..=MAX_SHIFT);

    if shift_x == 0 && shift_y == 0 {
        target_buffer.copy_from_slice(src_image);
        return;
    }

    const W: i32 = IMAGE_WIDTH as i32;
    const H: i32 = IMAGE_HEIGHT as i32;
    let y_start = 0.max(-shift_y) as usize;
    let y_end = H.min(H - shift_y) as usize;
    let x_start = 0.max(-shift_x) as usize;
    let x_end = W.min(W - shift_x) as usize;
    let copy_width = x_end - x_start;

    target_buffer.fill(0.0);

    for y in y_start..y_end {
        let src_y = (y as i32 + shift_y) as usize;
        let src_x = (x_start as i32 + shift_x) as usize;
        let dst_idx = y * IMAGE_WIDTH + x_start;
        let src_idx = src_y * IMAGE_WIDTH + src_x;
        target_buffer[dst_idx..dst_idx + copy_width]
            .copy_from_slice(&src_image[src_idx..src_idx + copy_width]);
    }
}

fn train_epoch(
    cfg: &Config,
    model: &mut FFModel,
    images: &[[f32; NPIXELS]],
    labels: &[u8],
    epoch: usize,
    rng: &mut SmallRng,
    ws: &mut BatchWorkspace,
    indices: &mut [usize],
    seed_buffer: &mut [u64],
) -> f32 {
    let num_batches = images.len() / cfg.batch_size;
    let epsgain = if epoch < MAX_EPOCH / 2 {
        1.0
    } else {
        (1.0 + 2.0 * (MAX_EPOCH - epoch) as f32) / MAX_EPOCH as f32
    };

    let mut total_cost = 0.0;
    indices.shuffle(rng);

    for batch_idx in 0..num_batches {
        // --- 0. PREPARE POSITIVE BATCH (POSITIVE) ---
        // Prepares the batch in parallel.
        // Generate seeds for this batch on the main thread for determinism
        for seed in seed_buffer.iter_mut() {
            *seed = rng.next_u64();
        }

        let batch_start = batch_idx * cfg.batch_size;
        let chunk_indices = &indices[batch_start..batch_start + cfg.batch_size];

        ws.data
            .data
            .par_chunks_exact_mut(NPIXELS)
            .zip(ws.targets.data.par_chunks_exact_mut(NUMLAB))
            .zip(chunk_indices)
            .zip(&*seed_buffer)
            .for_each(|(((img_buf, target_buf), &sample_idx), seed)| {
                let mut local_rng = SmallRng::seed_from_u64(*seed);

                let img_buf: &mut [f32] = img_buf;
                let target_buf: &mut [f32] = target_buf;
                let label = labels[sample_idx] as usize;

                // 1. Augment / Copy
                if USE_AUGMENTATION {
                    apply_random_shift(&images[sample_idx], img_buf, &mut local_rng);
                } else {
                    img_buf.copy_from_slice(&images[sample_idx][..]);
                }

                // 2. Set Targets (One Hot)
                target_buf.fill(0.0);
                target_buf[label] = 1.0;

                // 3. Embed Label
                img_buf[..NUMLAB].fill(0.0);
                img_buf[label] = LABELSTRENGTH;
            });

        // Initialize first layer
        ws.pos_nst[0].data.copy_from_slice(&ws.data.data);
        ws.pos_nst[0].norm_rows();

        // --- 1. FORWARD PASS (POSITIVE) ---
        for l in 0..model.layers.len() {
            let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
            layer_io_into(
                cfg,
                &prev_nst[l],
                &model.layers[l],
                &mut ws.pos_st[l],
                &mut next_nst[0],
                Some(rng),
            );
            sanitise_slice(&mut ws.pos_st[l].data);

            //let cols = ws.pos_st[l].cols;
            //for r in 0..cfg.batch_size {
            //    ws.pos_probs[l][r] = goodness(&ws.pos_st[l].data[r * cols..(r + 1) * cols], TEMP);
            //}
            update_batch_goodness(&ws.pos_st[l], TEMP, &mut ws.pos_probs[l]);
        }

        // --- 2. SOFTMAX ---
        ws.lab_data.data.copy_from_slice(&ws.data.data);
        for r in 0..cfg.batch_size {
            ws.lab_data.data[r * NPIXELS..r * NPIXELS + NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);
        }
        ws.softmax_nst[0].data.copy_from_slice(&ws.lab_data.data);
        ws.softmax_nst[0].norm_rows();

        for l in 0..model.layers.len() {
            let (prev_nst, next_nst) = ws.softmax_nst.split_at_mut(l + 1);
            layer_io_into(
                cfg,
                &prev_nst[l],
                &model.layers[l],
                &mut ws.softmax_st[l],
                &mut next_nst[0],
                Some(rng),
            );
        }

        // Supervised Contributions
        ws.labin.data.fill(0.0);
        for l in MINLEVELSUP - 1..model.layers.len() {
            if let Some(sw) = &model.layers[l].supweights {
                ws.softmax_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
                for i in 0..ws.labin.data.len() {
                    ws.labin.data[i] += ws.sup_contrib.data[i];
                }
            }
        }
        sanitise_slice(&mut ws.labin.data);

        // Softmax & Gradients
        for r in 0..cfg.batch_size {
            let row = &mut ws.labin.data[r * NUMLAB..(r + 1) * NUMLAB];
            let target_row = &ws.targets.data[r * NUMLAB..(r + 1) * NUMLAB];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = row
                .iter_mut()
                .map(|x| {
                    *x = (*x - max_val).exp();
                    *x
                })
                .sum();

            let mut correct_p = 0.0;
            for c in 0..NUMLAB {
                row[c] /= sum_exp;
                correct_p += row[c] * target_row[c];
                ws.dc_din_sup.data[r * NUMLAB + c] = target_row[c] - row[c];
            }
            total_cost += -(correct_p + TINY).ln();
        }

        for (l, layer) in model.layers.iter_mut().enumerate().skip(MINLEVELSUP - 1) {
            if let Some(sw) = &mut layer.supweights {
                ws.softmax_nst[l + 1].t_matmul_into(&ws.dc_din_sup, &mut ws.sw_grad_tmp);
                let g_buf = layer.sup_weight_velocity.as_mut().unwrap();
                let scale = epsgain * EPSILONSUP;
                for i in 0..sw.data.len() {
                    g_buf.data[i] = DELAY * g_buf.data[i]
                        + (1.0 - DELAY) * ws.sw_grad_tmp.data[i] / cfg.batch_size as f32;
                    sw.data[i] += scale * (g_buf.data[i] - SUPWC * sw.data[i]);
                }
            }
        }

        // --- 3. NEGATIVE PASS ---
        ws.neg_data.data.copy_from_slice(&ws.data.data);
        for r in 0..cfg.batch_size {
            let start_idx = r * NUMLAB;
            let probs = &ws.labin.data[start_idx..start_idx + NUMLAB];
            let targets = &ws.targets.data[start_idx..start_idx + NUMLAB];

            // Use "Fitness Proportionate Selection" to select a difficult non-target
            let sum_non_target: f32 = probs
                .iter()
                .zip(targets)
                .filter(|&(_, &t)| t == 0.0)
                .map(|(&p, _)| p)
                .sum();

            let find_neg = |rv: f32, sum_non_target: f32| -> usize {
                let mut cum = 0.0;
                for (c, (&p, &t)) in probs.iter().zip(targets).enumerate() {
                    if t == 0.0 {
                        cum += p / (sum_non_target + TINY);
                        if rv < cum {
                            return c;
                        }
                    }
                }
                // fell through - happens if target is close to 1 (sum_non_target close to 0)
                targets.iter().position(|&t| t == 0.0).unwrap_or(0)
            };
            let sel = find_neg(rng.random(), sum_non_target);

            let img_start = r * NPIXELS;
            let label_slice = &mut ws.neg_data.data[img_start..img_start + NUMLAB];
            label_slice.fill(0.0);
            label_slice[sel] = LABELSTRENGTH;
        }

        ws.neg_nst[0].data.copy_from_slice(&ws.neg_data.data);
        ws.neg_nst[0].norm_rows();

        // --- 4. WEIGHT UPDATES ---
        for (l, layer) in model.layers.iter_mut().enumerate() {
            let cols = layer.weights.cols;
            let layer_mean: f32 = layer.activity_running_mean.iter().sum::<f32>() / cols as f32;
            let inv_bs = 1.0 / cfg.batch_size as f32;

            for r in 0..cfg.batch_size {
                let p = ws.pos_probs[l][r];
                let row_offset = r * cols;
                for c in 0..cols {
                    let st = ws.pos_st[l].data[row_offset + c];
                    layer.activity_running_mean[c] =
                        0.9 * layer.activity_running_mean[c] + 0.1 * (st * inv_bs);
                    let reg = cfg.lambda_mean * (layer_mean - layer.activity_running_mean[c]);
                    ws.pos_dc_din[l].data[row_offset + c] = (1.0 - p) * st + reg;
                }
            }
            ws.pos_nst[l].t_matmul_into(&ws.pos_dc_din[l], &mut ws.pos_dw[l]);

            let (prev_nst, next_nst) = ws.neg_nst.split_at_mut(l + 1);
            layer_io_into(
                cfg,
                &prev_nst[l],
                layer,
                &mut ws.neg_st[l],
                &mut next_nst[0],
                Some(rng),
            );

            for r in 0..cfg.batch_size {
                let row_offset = r * cols;
                let row = &ws.neg_st[l].data[row_offset..row_offset + cols];
                let p_neg = goodness(row, TEMP);
                for c in 0..cols {
                    ws.neg_dc_din[l].data[row_offset + c] = -p_neg * row[c];
                }
            }
            ws.neg_nst[l].t_matmul_into(&ws.neg_dc_din[l], &mut ws.neg_dw[l]);

            let w_scale = epsgain * EPSILON;
            let wg = &mut layer.weight_velocity.data;
            let w = &mut layer.weights.data;
            let pdw = &ws.pos_dw[l].data;
            let ndw = &ws.neg_dw[l].data;

            // Weight Update (Vectorised & Parallel)
            wg.par_iter_mut()
                .zip(w.par_iter_mut())
                .zip(pdw.par_iter())
                .zip(ndw.par_iter())
                .for_each(|(((wg_i, w_i), pdw_i), ndw_i)| {
                    let g = (pdw_i + ndw_i) * inv_bs;
                    *wg_i = DELAY * *wg_i + (1.0 - DELAY) * g;
                    *w_i += w_scale * (*wg_i - WC * *w_i);
                });

            // Weight Update
            //for i in 0..w.len() {
            //    let g = (pdw[i] + ndw[i]) * inv_bs;
            //    wg[i] = DELAY * wg[i] + (1.0 - DELAY) * g;
            //    w[i] += w_scale * (wg[i] - WC * w[i]);
            //}

            // Bias Update (Vectorised & Parallel)
            layer
                .biases_grad
                .par_iter_mut()
                .zip(layer.biases.par_iter_mut())
                .enumerate()
                .for_each(|(c, (bg_c, b_c))| {
                    let g: f32 = (0..cfg.batch_size)
                        .map(|r| {
                            ws.pos_dc_din[l].data[r * cols + c]
                                + ws.neg_dc_din[l].data[r * cols + c]
                        })
                        .sum();
                    *bg_c = DELAY * (*bg_c) + (1.0 - DELAY) * g * inv_bs;
                    *b_c += w_scale * (*bg_c);
                });

            // Bias Update
            //let bg = &mut layer.biases_grad;
            //let b = &mut layer.biases;
            //for c in 0..b.len() {
            //    let mut g = 0.0;
            //    for r in 0..cfg.batch_size {
            //        g += ws.pos_dc_din[l].data[r * cols + c] + ws.neg_dc_din[l].data[r * cols + c];
            //    }
            //    bg[c] = DELAY * bg[c] + (1.0 - DELAY) * (g * inv_bs);
            //    b[c] += w_scale * bg[c];
            //}
        }
    }
    total_cost / num_batches as f32
}

fn predict(cfg: &Config, model: &FFModel, image: &[f32], ws: &mut BatchWorkspace) -> usize {
    ws.data.data[..NPIXELS].copy_from_slice(image);
    ws.data.data[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);
    let input_len = ws.pos_nst[0].data.len();
    ws.pos_nst[0]
        .data
        .copy_from_slice(&ws.data.data[..input_len]);
    ws.pos_nst[0].norm_rows();

    for l in 0..model.layers.len() {
        let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
        layer_io_into(
            cfg,
            &prev_nst[l],
            &model.layers[l],
            &mut ws.pos_st[l],
            &mut next_nst[0],
            None,
        );
    }

    let mut scores = [0.0f32; NUMLAB];
    for l in MINLEVELSUP - 1..model.layers.len() {
        if let Some(sw) = &model.layers[l].supweights {
            ws.pos_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
            for c in 0..NUMLAB {
                scores[c] += ws.sup_contrib.data[c];
            }
        }
    }
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn fftest(
    cfg: &Config,
    model: &FFModel,
    images: &[[f32; NPIXELS]],
    labels: &[u8],
) -> (usize, usize) {
    let errors: usize = images
        .par_iter()
        .zip(labels)
        .map_init(
            || BatchWorkspace::new(&LAYERS, 1),
            |ws, (img, &label)| {
                if predict(cfg, model, img, ws) != label as usize {
                    1
                } else {
                    0
                }
            },
        )
        .sum();
    (errors, images.len())
}

fn train_model(
    cfg: &Config,
    train_imgs: &[[f32; NPIXELS]],
    val_imgs: &[[f32; NPIXELS]],
    train_labels: &[u8],
    val_labels: &[u8],
) -> Result<(), MnistError> {
    // UNIFIED RNG: Single SmallRng used for everything.
    let mut rng = SmallRng::seed_from_u64(1234);

    let mut model = FFModel::new(&mut rng);

    let mut ws = BatchWorkspace::new(&LAYERS, cfg.batch_size);

    // Initialize indices specifically for the training slice size
    let mut indices: Vec<usize> = (0..train_imgs.len()).collect();
    let mut seed_buffer: Vec<u64> = vec![0; cfg.batch_size];

    for epoch in 0..MAX_EPOCH {
        let cost = train_epoch(
            cfg,
            &mut model,
            train_imgs,
            train_labels,
            epoch,
            &mut rng,
            &mut ws,
            &mut indices,
            &mut seed_buffer,
        );

        if (epoch > 0 && epoch % 5 == 0) || epoch == MAX_EPOCH - 1 {
            let (errors0, total0) = fftest(cfg, &model, train_imgs, train_labels);
            let (errors1, total1) = fftest(cfg, &model, val_imgs, val_labels);
            println!(
                "Epoch {epoch:3} | Cost: {cost:8.4} | Err Train: ({errors0}/{total0}), Err Val: ({errors1}/{total1})"
            );
        } else {
            println!("Epoch {epoch:3} | Cost: {cost:8.4}");
        }
    }
    model.save("model_ff.bin")?;
    Ok(())
}

fn calc_confusions(
    cfg: &Config,
    model: &FFModel,
    images: &[[f32; NPIXELS]],
    labels: &[u8],
) -> [[usize; 10]; 10] {
    let mut matrix = [[0usize; 10]; 10];
    let mut ws = BatchWorkspace::new(&LAYERS, 1);
    println!("Calculating confusion matrix...");
    for (img, &label) in images.iter().zip(labels) {
        let pred = predict(cfg, model, img, &mut ws);
        matrix[label as usize][pred] += 1;
    }
    matrix
}

fn print_confusions(matrix: &[[usize; 10]; 10]) {
    println!("\nConfusion Matrix (Actual vs Predicted):\n");
    print!("Actual |");
    for i in 0..10 {
        print!("  P{i}  ");
    }
    println!("\n-------|------------------------------------------------------------");
    for i in 0..10 {
        print!("  A{}   |", i);
        for j in 0..10 {
            let count = matrix[i][j];
            if count == 0 {
                print!("{:>5} ", ".");
            } else {
                print!("{count:>5} ");
            }
        }
        println!();
    }
    println!("--------------------------------------------------------------------");
}

fn main() -> Result<(), MnistError> {
    let cfg = Config::parse();

    println!("Training Forward-Forward Model on {} ...", cfg.dir);
    let data = Mnist::load(&cfg.dir)?;
    let train_imgs: Vec<[f32; NPIXELS]> = data
        .train_images
        .iter()
        .map(|img| img.as_f32_array())
        .collect();
    let test_imgs: Vec<[f32; NPIXELS]> = data
        .test_images
        .iter()
        .map(|img| img.as_f32_array())
        .collect();
    let train_val_split = 50000;
    let (train_imgs, val_imgs) = train_imgs.split_at(train_val_split);
    let (train_labels, val_labels) = data.train_labels.split_at(train_val_split);

    train_model(&cfg, train_imgs, val_imgs, train_labels, val_labels)?;
    let model = FFModel::load("model_ff.bin")?;
    let (test_errors, test_total) = fftest(&cfg, &model, &test_imgs, &data.test_labels);
    let (train_errors, train_total) = fftest(&cfg, &model, train_imgs, train_labels);
    let (val_errors, val_total) = fftest(&cfg, &model, val_imgs, val_labels);
    println!(
        "Errors - Test: {test_errors}/{test_total}, Val: {val_errors}/{val_total}, Train: {train_errors}/{train_total}"
    );
    let matrix = calc_confusions(&cfg, &model, &test_imgs, &data.test_labels);
    print_confusions(&matrix);
    Ok(())
}
