use clap::Parser;
use forward_forward::Mat;
use mnist::{IMAGE_HEIGHT, IMAGE_WIDTH, Mnist, NPIXELS, NUM_LABELS, error::MnistError};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// --- Constants ---
const USE_DROPOUT: bool = true;
const USE_AUGMENTATION: bool = true;
const AUG_MAX_SHIFT: i32 = 1; // AUGMENTATION - PIXELS to shift by
const SANITISE: bool = false;
const TINY: f32 = 1e-10; // For numerical stability
const MINLEVELSUP: usize = 2;
const LAYERS: [usize; 4] = [784, 1000, 1000, 1000];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Config {
    #[arg(long, default_value_t = 0)]
    /// Number of threads to use (0 = auto)
    pub threads: usize,

    #[arg(long)]
    /// Path to a saved model. If provided, training is skipped.
    pub model: Option<String>,

    #[arg(short, long="dir", default_value_t = String::from("MNIST"))]
    /// X-axis range: min,max
    dir: String,

    #[arg(long, default_value_t = 1234)]
    /// Seed used to initialise the PRNG
    pub seed: u64,

    #[arg(long, default_value_t = 0.15)]
    /// Probability of zeroing activations to prevent over-fitting
    pub dropout: f32,

    #[arg(long, default_value_t = 0.03)]
    /// Strength of the penalty for neurons straying from the target mean activation
    // Peer normalization: we regress the mean activity of each neuron towards the average mean for its layer.
    // This prevents dead or hysterical units. We pretend there is a gradient even when hidden units are off.
    // Choose strength of regression (LAMBDAMEAN) so that average activities are similar but not too similar.
    pub lambda_mean: f32,

    //// Using Vec allows flexible network depth, unlike [usize; 4]
    //#[arg(long, value_delimiter = ',', default_value = "784,1000,1000,1000")]
    //pub layers: Vec<usize>,
    #[arg(long, default_value_t = 100)]
    pub batch_size: usize,

    /// Weight decay (L2 regularization) for main layers
    #[arg(long, default_value_t = 0.002)]
    pub weight_decay: f32,

    /// Weight decay (L2 regularization) for supervised layers
    #[arg(long, default_value_t = 0.003)]
    pub sup_weight_decay: f32,

    #[arg(long, default_value_t = 0.9)]
    /// Momentum factor (exponential moving average of gradients)
    pub momentum: f32,

    /// Learning rate (step size) for weight updates
    #[arg(short = 'e', long = "lr", default_value_t = 0.01)]
    pub epsilon: f32,

    /// Learning rate for the supervised output weights
    #[arg(long = "lr-sup", default_value_t = 0.1)]
    pub epsilon_sup: f32,

    #[arg(long, default_value_t = 1.0)]
    /// Magnitude of the label embedding in the input layer
    pub label_strength: f32,

    #[arg(long, default_value_t = 1.0)]
    /// Temperature scaling for the goodness function (controls sigmoid sharpness)
    pub temperature: f32,

    #[arg(long, default_value_t = 200)]
    /// Number of times to cycle through the training data.
    pub max_epoch: usize,
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

impl Layer {
    /// Applies MatMul -> Bias -> ReLU -> Dropout -> Normalization
    fn forward(
        &self,
        cfg: &Config,
        vin: &Mat,
        raw_activations: &mut Mat,
        normalised_activations: &mut Mat,
        orng: Option<&mut SmallRng>,
    ) {
        // 1. Standard Linear Projection
        vin.matmul_into(&self.weights, raw_activations);
        let cols = raw_activations.cols;

        // 2. Element-wise Bias + ReLU (Parallelizable)
        raw_activations.data.par_chunks_mut(cols).for_each(|row| {
            for (val, &bias) in row.iter_mut().zip(self.biases.iter()) {
                *val = (*val + bias).max(0.0); // ReLU
            }
        });

        // 3. NORMALIZE FIRST
        // We copy raw ReLU output to normalised_activations then normalize it.
        // This ensures the NEXT layer gets unit-length vectors.
        normalised_activations
            .data
            .copy_from_slice(&raw_activations.data);
        normalised_activations.norm_rows();

        // 4. APPLY DROPOUT TO BOTH (if training)
        // We apply dropout to raw_activations so the "Goodness" (Energy)
        // calculation is actually affected by the missing neurons.
        if let Some(rng) = orng
            && USE_DROPOUT
        {
            let dropout_scale: f32 = 1.0 / (1.0 - cfg.dropout);

            for i in 0..raw_activations.data.len() {
                if rng.random::<f32>() < cfg.dropout {
                    raw_activations.data[i] = 0.0;
                    normalised_activations.data[i] = 0.0;
                } else {
                    raw_activations.data[i] *= dropout_scale;
                    normalised_activations.data[i] *= dropout_scale;
                }
            }
        }
    }
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
                    supweights: Some(Mat::zeros(fanout, NUM_LABELS)),
                    weight_velocity: Mat::zeros(fanin, fanout),
                    biases_grad: vec![0.0; fanout],
                    sup_weight_velocity: Some(Mat::zeros(fanout, NUM_LABELS)),
                    activity_running_mean: vec![0.5; fanout],
                }
            })
            .collect();
        FFModel { layers }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&(self.layers.len() as u64).to_le_bytes())?;

        for layer in &self.layers {
            layer.weights.write_raw(&mut writer)?;

            writer.write_all(&(layer.biases.len() as u64).to_le_bytes())?;

            for &bias in &layer.biases {
                writer.write_all(&bias.to_le_bytes())?;
            }

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

        let mut u64_buf = [0u8; 8];
        let mut f32_buf = [0u8; 4];
        reader.read_exact(&mut u64_buf)?;
        let num_layers = u64::from_le_bytes(u64_buf) as usize;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Mat::read_raw reads the rows/cols, so we recover topology automatically
            let weights = Mat::read_raw(&mut reader)?;

            reader.read_exact(&mut u64_buf)?;
            let b_len = u64::from_le_bytes(u64_buf) as usize;
            let mut biases = Vec::with_capacity(b_len);
            for _ in 0..b_len {
                reader.read_exact(&mut f32_buf)?;
                biases.push(f32::from_le_bytes(f32_buf));
            }

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

    fn run_forward(
        &self,
        cfg: &Config,
        input: &Mat,
        st_workspace: &mut [Mat],
        nst_workspace: &mut [Mat],
        mut rng: Option<&mut SmallRng>,
    ) {
        nst_workspace[0].data.copy_from_slice(&input.data);
        nst_workspace[0].norm_rows();

        for l in 0..self.layers.len() {
            let (prev_nst, next_nst) = nst_workspace.split_at_mut(l + 1);
            self.layers[l].forward(
                cfg,
                &prev_nst[l],
                &mut st_workspace[l],
                &mut next_nst[0],
                // Re-wrapping the Option to satisfy the borrow checker
                rng.as_mut().map(|r| r as &mut SmallRng),
                //rng,
            );
            sanitise_slice(&mut st_workspace[l].data);
        }
    }

    fn predict(&self, cfg: &Config, image: &[f32], ws: &mut BatchWorkspace) -> usize {
        ws.data.data[..NPIXELS].copy_from_slice(image);
        ws.data.data[..NUM_LABELS].fill(cfg.label_strength / NUM_LABELS as f32);

        self.run_forward(cfg, &ws.data, &mut ws.pos_st, &mut ws.pos_nst, None);

        let mut scores = [0.0f32; NUM_LABELS];
        for l in MINLEVELSUP - 1..self.layers.len() {
            if let Some(sw) = &self.layers[l].supweights {
                ws.pos_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
                for c in 0..NUM_LABELS {
                    scores[c] += ws.sup_contrib.data[c];
                }
            }
        }

        argmax(&scores)
    }

    fn predict_energy(&self, cfg: &Config, image: &[f32], ws: &mut BatchWorkspace) -> usize {
        let mut label_energies = [0.0f32; NUM_LABELS];

        for lab in 0..NUM_LABELS {
            // Prepare input for this specific label
            ws.data.data[..NPIXELS].copy_from_slice(image);
            set_one_hot(&mut ws.data.data[..NUM_LABELS], lab, cfg.label_strength);

            self.run_forward(cfg, &ws.data, &mut ws.pos_st, &mut ws.pos_nst, None);

            // Accumulate Energy (Sum of Squares) from MINLEVELSUP onwards
            // In energy tests, we use the raw states (pos_st)
            for l in (MINLEVELSUP - 1)..self.layers.len() {
                let row = &ws.pos_st[l].data; // Batch size is 1 in fftest
                let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
                label_energies[lab] += sum_sq;
            }
        }

        argmax(&label_energies)
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
            targets: Mat::zeros(batch_size, NUM_LABELS),
            lab_data: Mat::zeros(batch_size, layers[0]),
            labin: Mat::zeros(batch_size, NUM_LABELS),
            dc_din_sup: Mat::zeros(batch_size, NUM_LABELS),
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
            sup_contrib: Mat::zeros(batch_size, NUM_LABELS),
            sw_grad_tmp: Mat::zeros(*layers.iter().max().unwrap(), NUM_LABELS),
        }
    }
}

// --- Helper Functions ---

#[inline(always)]
fn argmax(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

#[inline(always)]
fn set_one_hot(slice: &mut [f32], index: usize, value: f32) {
    slice[..NUM_LABELS].fill(0.0);
    slice[index] = value;
}

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

fn apply_random_shift(src_image: &[f32; NPIXELS], target_buffer: &mut [f32], rng: &mut SmallRng) {
    let shift_x = rng.random_range(-AUG_MAX_SHIFT..=AUG_MAX_SHIFT);
    let shift_y = rng.random_range(-AUG_MAX_SHIFT..=AUG_MAX_SHIFT);

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
    let lr_scale = forward_forward::get_lr_scale(epoch, cfg.max_epoch);

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
            .zip(ws.targets.data.par_chunks_exact_mut(NUM_LABELS))
            .zip(chunk_indices)
            .zip(&*seed_buffer)
            .for_each(|(((img_buf, target_buf), &sample_idx), seed)| {
                let mut local_rng = SmallRng::seed_from_u64(*seed);

                let label = labels[sample_idx] as usize;

                // 1. Augment / Copy
                if USE_AUGMENTATION {
                    apply_random_shift(&images[sample_idx], img_buf, &mut local_rng);
                } else {
                    img_buf.copy_from_slice(&images[sample_idx][..]);
                }

                // 2. Set Targets (One Hot)
                set_one_hot(target_buf, label, 1.0);

                // 3. Embed Label
                set_one_hot(img_buf, label, cfg.label_strength);
            });

        // Initialize first layer
        ws.pos_nst[0].data.copy_from_slice(&ws.data.data);
        ws.pos_nst[0].norm_rows();

        // --- 1. FORWARD PASS (POSITIVE) ---
        for l in 0..model.layers.len() {
            let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
            model.layers[l].forward(
                cfg,
                &prev_nst[l],
                &mut ws.pos_st[l],
                &mut next_nst[0],
                Some(rng),
            );
            sanitise_slice(&mut ws.pos_st[l].data);

            //let cols = ws.pos_st[l].cols;
            //for r in 0..cfg.batch_size {
            //    ws.pos_probs[l][r] = goodness(&ws.pos_st[l].data[r * cols..(r + 1) * cols], cfg.temperature);
            //}
            update_batch_goodness(&ws.pos_st[l], cfg.temperature, &mut ws.pos_probs[l]);
        }

        // --- 2. SOFTMAX ---
        ws.lab_data.data.copy_from_slice(&ws.data.data);
        for r in 0..cfg.batch_size {
            ws.lab_data.data[r * NPIXELS..r * NPIXELS + NUM_LABELS]
                .fill(cfg.label_strength / NUM_LABELS as f32);
        }
        ws.softmax_nst[0].data.copy_from_slice(&ws.lab_data.data);
        ws.softmax_nst[0].norm_rows();

        for l in 0..model.layers.len() {
            let (prev_nst, next_nst) = ws.softmax_nst.split_at_mut(l + 1);
            model.layers[l].forward(
                cfg,
                &prev_nst[l],
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
            let row = &mut ws.labin.data[r * NUM_LABELS..(r + 1) * NUM_LABELS];
            let target_row = &ws.targets.data[r * NUM_LABELS..(r + 1) * NUM_LABELS];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = row
                .iter_mut()
                .map(|x| {
                    *x = (*x - max_val).exp();
                    *x
                })
                .sum();

            let mut correct_p = 0.0;
            for c in 0..NUM_LABELS {
                row[c] /= sum_exp;
                correct_p += row[c] * target_row[c];
                ws.dc_din_sup.data[r * NUM_LABELS + c] = target_row[c] - row[c];
            }
            total_cost += -(correct_p + TINY).ln();
        }

        for (l, layer) in model.layers.iter_mut().enumerate().skip(MINLEVELSUP - 1) {
            if let Some(sw) = &mut layer.supweights {
                ws.softmax_nst[l + 1].t_matmul_into(&ws.dc_din_sup, &mut ws.sw_grad_tmp);
                let g_buf = layer.sup_weight_velocity.as_mut().unwrap();
                let scale = lr_scale * cfg.epsilon_sup;
                for i in 0..sw.data.len() {
                    g_buf.data[i] = cfg.momentum * g_buf.data[i]
                        + (1.0 - cfg.momentum) * ws.sw_grad_tmp.data[i] / cfg.batch_size as f32;
                    sw.data[i] += scale * (g_buf.data[i] - cfg.sup_weight_decay * sw.data[i]);
                }
            }
        }

        // --- 3. NEGATIVE PASS ---
        ws.neg_data.data.copy_from_slice(&ws.data.data);
        for r in 0..cfg.batch_size {
            let start_idx = r * NUM_LABELS;
            let probs = &ws.labin.data[start_idx..start_idx + NUM_LABELS];
            let targets = &ws.targets.data[start_idx..start_idx + NUM_LABELS];

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
            let label_slice = &mut ws.neg_data.data[img_start..img_start + NUM_LABELS];
            set_one_hot(label_slice, sel, cfg.label_strength);
        }

        ws.neg_nst[0].data.copy_from_slice(&ws.neg_data.data);
        ws.neg_nst[0].norm_rows();

        // --- 4. WEIGHT UPDATES ---
        for (l, layer) in model.layers.iter_mut().enumerate() {
            let cols = layer.weights.cols;
            let inv_bs = 1.0 / cfg.batch_size as f32;

            // Calculate the actual batch mean for each neuron
            let mut batch_means = vec![0.0; cols];
            for r in 0..cfg.batch_size {
                let row_offset = r * cols;
                for c in 0..cols {
                    batch_means[c] += ws.pos_st[l].data[row_offset + c] * inv_bs;
                }
            }

            // Update the running mean once per batch
            for c in 0..cols {
                layer.activity_running_mean[c] =
                    0.9 * layer.activity_running_mean[c] + 0.1 * batch_means[c];
            }

            // calculate the global layer mean for the regularization term
            let layer_mean: f32 = layer.activity_running_mean.iter().sum::<f32>() / cols as f32;

            // gradient calculations
            for r in 0..cfg.batch_size {
                let p = ws.pos_probs[l][r];
                let row_offset = r * cols;
                for c in 0..cols {
                    let st = ws.pos_st[l].data[row_offset + c];
                    let reg = cfg.lambda_mean * (layer_mean - layer.activity_running_mean[c]);
                    ws.pos_dc_din[l].data[row_offset + c] = (1.0 - p) * st + reg;
                    // This is a regularizer that encourages the average activity of a unit to match that for
                    // all the units in the layer. Notice that we do not gate by (states>0) for this extra term.
                    // This allows the extra term to revive units that are always off.  May not be needed.
                }
            }

            ws.pos_nst[l].t_matmul_into(&ws.pos_dc_din[l], &mut ws.pos_dw[l]);

            let (prev_nst, next_nst) = ws.neg_nst.split_at_mut(l + 1);
            layer.forward(
                cfg,
                &prev_nst[l],
                &mut ws.neg_st[l],
                &mut next_nst[0],
                Some(rng),
            );

            for r in 0..cfg.batch_size {
                let row_offset = r * cols;
                let row = &ws.neg_st[l].data[row_offset..row_offset + cols];
                let p_neg = goodness(row, cfg.temperature);
                for c in 0..cols {
                    ws.neg_dc_din[l].data[row_offset + c] = -p_neg * row[c];
                }
            }
            ws.neg_nst[l].t_matmul_into(&ws.neg_dc_din[l], &mut ws.neg_dw[l]);

            let w_scale = lr_scale * cfg.epsilon;
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
                    *wg_i = cfg.momentum * *wg_i + (1.0 - cfg.momentum) * g;
                    *w_i += w_scale * (*wg_i - cfg.weight_decay * *w_i);
                });

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
                    *bg_c = cfg.momentum * (*bg_c) + (1.0 - cfg.momentum) * g * inv_bs;
                    *b_c += w_scale * (*bg_c);
                });
        }
    }
    total_cost / num_batches as f32
}

#[derive(Debug, Copy, Clone)]
pub enum TestMode {
    Softmax,
    Energy,
}

fn fftest(
    cfg: &Config,
    model: &FFModel,
    images: &[[f32; NPIXELS]],
    labels: &[u8],
    mode: TestMode,
) -> (usize, usize) {
    let errors: usize = images
        .par_iter()
        .zip(labels)
        .map_init(
            || BatchWorkspace::new(&LAYERS, 1),
            |ws, (img, &label)| {
                let guess = match mode {
                    TestMode::Softmax => model.predict(cfg, img, ws),
                    TestMode::Energy => model.predict_energy(cfg, img, ws),
                };
                if guess != label as usize { 1 } else { 0 }
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
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut model = FFModel::new(&mut rng);
    let mut ws = BatchWorkspace::new(&LAYERS, cfg.batch_size);

    // Initialize indices specifically for the training slice size
    let mut indices: Vec<usize> = (0..train_imgs.len()).collect();
    let mut seed_buffer: Vec<u64> = vec![0; cfg.batch_size];

    for epoch in 0..cfg.max_epoch {
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

        if (epoch > 0 && epoch % 5 == 0) || epoch == cfg.max_epoch - 1 {
            let (errors0, total0) =
                fftest(cfg, &model, train_imgs, train_labels, TestMode::Softmax);
            let (errors1, total1) = fftest(cfg, &model, val_imgs, val_labels, TestMode::Softmax);
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
        let pred = model.predict(cfg, img, &mut ws);
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

    // Initialize Rayon if a specific count is requested
    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .map_err(|e| {
                eprintln!("Warning: Could not set thread count: {e}");
            })
            .ok();
        println!("Rayon initialized with {} threads", cfg.threads);
    } else {
        println!("Rayon using default thread count (logical cores)");
    }

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

    let model = if let Some(path) = &cfg.model {
        println!("Loading model from {path}...");
        FFModel::load(path)?
    } else {
        println!("Training Forward-Forward Model \n{cfg:?}");
        train_model(&cfg, train_imgs, val_imgs, train_labels, val_labels)?;
        FFModel::load("model_ff.bin").unwrap()
    };

    for mode in [TestMode::Softmax, TestMode::Energy] {
        println!("TestMode: {mode:?}");
        let (test_errors, test_total) = fftest(&cfg, &model, &test_imgs, &data.test_labels, mode);
        let (train_errors, train_total) = fftest(&cfg, &model, train_imgs, train_labels, mode);
        let (val_errors, val_total) = fftest(&cfg, &model, val_imgs, val_labels, mode);
        println!(
            "Errors - Test: {test_errors}/{test_total}, Val: {val_errors}/{val_total}, Train: {train_errors}/{train_total}"
        );
    }
    let matrix = calc_confusions(&cfg, &model, &test_imgs, &data.test_labels);
    print_confusions(&matrix);
    Ok(())
}
