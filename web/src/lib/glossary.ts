// Plain-language definitions: the main page's Glossary chapter shows GLOSSARY; the explorers' <Term> tooltips use TERMS.
export const GLOSSARY: [string, string][] = [
  ["Classifier", "A program that looks at an input (here, a picture) and picks a label from a fixed list."],
  ["Prediction set", "A list of possible labels instead of a single answer. Bigger list = more doubt."],
  ["Coverage", "The fraction of pictures whose true label is inside the set. We target 90%."],
  ["α (alpha)", "The error rate we accept. α = 0.1 means the set may miss the truth 10% of the time."],
  ["Calibration set", "Fresh labelled pictures, never used for training, used only to choose the cut-off."],
  ["Surprise (nonconformity) score", "How badly a label fits a picture, according to the model. For the simple rule: 1 − probability."],
  ["Cut-off q̂ (quantile)", "The score below which 90% of calibration scores fall. Labels scoring under it go into the set."],
  ["Exchangeable", "“New pictures are like the old ones”: shuffling the order of the pictures would not change anything statistically. The guarantee needs this."],
  ["Class-conditional coverage", "Coverage measured for one class at a time (only cats, only dogs…). The average can hide a low one."],
  ["Mondrian / per-class", "Using a separate cut-off for each class so each class aims for 90%."],
  ["PAC (safety margin)", "“Probably approximately correct”: choose a stricter cut-off so the target is met with high probability, not just on average."],
  ["Beta distribution", "The exact law for how coverage varies from one calibration set to another. Our worst-class formula is built from it."],
  ["Temperature scaling", "Dividing the model’s raw scores by a number T so its probabilities are less over-confident."],
  ["Distribution shift", "When new data differs from the training/calibration data (noise, fog, a different mix of classes)."],
];

/** Extra terms the explorers use, on top of the chapter glossary. */
const EXTRA: [string, string][] = [
  ["APS", "Adaptive Prediction Sets: add labels from most to least likely until enough probability is covered."],
  ["RAPS", "Regularised APS: APS plus a penalty for every label past the first few, so sets stay short."],
  ["Clustered", "Group classes that behave alike and give each group its own cut-off: between one cut-off and one per class."],
  ["ECE", "Expected calibration error: the average gap between how sure the model says it is and how often it is right."],
  ["ACI", "Adaptive conformal inference: nudges α up or down after every picture so coverage recovers when data drifts."],
  ["OOD", "Out of distribution: pictures of things the model was never trained on."],
  ["AUROC", "How well a score separates two groups: 50% is a coin flip, 100% is perfect."],
  ["Mahalanobis distance", "How far a picture's internal features are from the typical features of a class, in units of that class's spread."],
  ["Bootstrap CI", "A range for a difference, found by resampling the seeds many times; if it excludes 0, the difference is real."],
  ["Label shift", "The mix of classes changes (more cats, fewer trucks) while each class still looks the same."],
  ["Singleton", "A set with exactly one label: a normal, confident answer."],
  ["SSCV", "Size-stratified coverage violation: the biggest gap between coverage and target among sets of the same size."],
  ["Seed", "One full repeat of the experiment with a different random split and training run."],
];

const key = (t: string) => t.toLowerCase().replace(/\s*\(.*\)$/, "").split(" / ")[0];
export const TERMS: Record<string, string> = Object.fromEntries([...GLOSSARY, ...EXTRA].map(([t, d]) => [key(t), d]));
