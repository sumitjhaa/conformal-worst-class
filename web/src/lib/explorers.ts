// Every explorer: route slug, names, the main-page chapter it extends, and its guided-tour steps.
// The tour points at elements carrying data-tour="<key>" inside the explorer.
export type Tour = { at: string; text: string }[];
export type ExplorerInfo = { slug: string; short: string; title: string; blurb: string; group: string; chapter?: string; tour: Tour };

const T = (...steps: [string, string][]): Tour => steps.map(([at, text]) => ({ at, text }));

export const GROUPS = ["Basics", "Who it fails", "Under shift", "Under the hood", "Play"] as const;

export const EXPLORERS: ExplorerInfo[] = [
  { slug: "set-lab", short: "Set lab", group: "Basics", chapter: "try", title: "Build prediction sets live",
    blurb: "Pick a dataset, a rule and α, or drag the cut-off yourself, and watch coverage and set size respond on 5,000 real test pictures.",
    tour: T(["controls", "Choose the dataset, the rule and α here. Everything below is recomputed in your browser."], ["cutoff", "Override the calibrated cut-off and watch coverage fall below or rise above the target."], ["curve", "Each dot is one α: lower α means a higher target and bigger sets."], ["gallery", "Click any picture to open it in the per-image inspector."]) },
  { slug: "leaderboard", short: "Leaderboard", group: "Basics", chapter: "results", title: "Every method, every dataset",
    blurb: "Sort 14 conformal methods by coverage, set size or worst class, averaged over seeds, and test any difference with a bootstrap interval.",
    tour: T(["controls", "Pick the dataset and α. Numbers are means over all seeds, ± one standard deviation."], ["table", "Click a column header to sort. Rows under target coverage are marked."], ["pair", "Pick two methods to see whether their difference is real (the interval excludes 0) or noise."]) },
  { slug: "inspector", short: "Inspector", group: "Basics", chapter: "how", title: "Why did this label get in?",
    blurb: "Open one picture: its probabilities, each label's score next to the cut-off it must beat, and the set every rule gives it.",
    tour: T(["picker", "Pick a picture from the strip."], ["scores", "A label joins the set when its score bar ends left of the cut-off line."], ["rules", "The same picture under every rule: compare who gets in."]) },
  { slug: "left-out", short: "Who's left out", group: "Who it fails", chapter: "fair", title: "Which classes miss the promise",
    blurb: "Coverage class by class, by picture difficulty, and the spread of set sizes: where the 90% average hides much lower numbers.",
    tour: T(["classes", "One dot per class. The average is 90%, but look at the lowest dots."], ["deciles", "Pictures sorted from easiest to hardest: hard pictures are covered least."], ["sizes", "How many labels the sets have."]) },
  { slug: "worst-class", short: "Worst-class law", group: "Who it fails", chapter: "predict", title: "Predicting the worst class before testing",
    blurb: "The project's exact law for the lowest class coverage, checked against every seed, and a calculator for any number of classes and calibration pictures.",
    tour: T(["law", "Set the number of classes and calibration pictures per class: the curve is where the worst class will land."], ["seeds", "Each dot is a real run; the band is the law's 10–90% prediction."], ["ncal", "More calibration pictures per class pull the worst class up towards the target."]) },
  { slug: "simulator", short: "Coverage simulator", group: "Who it fails", chapter: "promise", title: "Re-run the experiment hundreds of times",
    blurb: "Reshuffle calibration and test pictures in your browser and see how coverage varies, against the exact Beta law and the guarantee's bounds.",
    tour: T(["controls", "Choose how many calibration pictures each run gets, then press Run."], ["hist", "Each bar counts reshuffles. The curve is the exact Beta law, the dashed lines the guarantee's bounds."]) },
  { slug: "grouping", short: "Grouping classes", group: "Who it fails", chapter: "fix", title: "One cut-off, one per class, or groups?",
    blurb: "How the number of class groups trades set size against the worst class, with and without the PAC safety margin.",
    tour: T(["chart", "Left: one shared cut-off. Right: one per class. Groups sit in between."]) },
  { slug: "shift", short: "Shift lab", group: "Under shift", chapter: "shift", title: "When new pictures look different",
    blurb: "Noise, blur, fog, a new test set and a new mix of classes: how far coverage falls, live on real corrupted pictures.",
    tour: T(["tabs", "Four kinds of shift: live corrupted pictures, CIFAR-10-C, synthetic noise and label shift."], ["live", "Pick a corruption: the sets below are recomputed on those pictures with the clean cut-off."]) },
  { slug: "aci", short: "Adaptive coverage", group: "Under shift", chapter: "shift", title: "Coverage that recovers over time",
    blurb: "A stream of pictures that gets noisier: watch fixed and adaptive (ACI) conformal react, step by step.",
    tour: T(["play", "Play the stream. The shaded bands are segments of increasing shift."], ["cov", "Rolling coverage: the fixed rule (lr 0) sinks, adaptive rules climb back."], ["alpha", "ACI lowers α_t when it misses, which grows the sets."]) },
  { slug: "ood", short: "Never-seen images", group: "Under shift", chapter: "shift", title: "Shown something it has never seen",
    blurb: "Feed the model pictures from another dataset: does the set size warn you? Empty, single and full sets, live.",
    tour: T(["stats", "Set-size behaviour on foreign pictures, recomputed live."], ["gallery", "Foreign pictures and the sets they get. Ideally these are large or empty."], ["table", "All methods, from the experiments: AUROC of “set is not a single label” as a warning signal."]) },
  { slug: "calibration", short: "Calibration lab", group: "Under the hood", chapter: "method", title: "Temperature and over-confidence",
    blurb: "Slide the temperature: the reliability diagram and ECE change, set sizes change, but coverage holds. Conformal does not need a calibrated model.",
    tour: T(["temp", "Drag T. Above 1 softens probabilities, below 1 sharpens them."], ["rel", "Bars on the diagonal mean “says 80%, right 80% of the time”."], ["sets", "Coverage stays near target at every T; only the set size pays."]) },
  { slug: "tuning", short: "Tuning & diagnostics", group: "Under the hood", chapter: "method", title: "Knobs the methods were tuned with",
    blurb: "The distance weight γ, tuned temperatures, the distance diagnostics and when empty sets appear.",
    tour: T(["gamma", "γ = 0 is the plain score; larger γ adds feature distance."], ["empty", "Empty sets appear when the model is more accurate than the target."]) },
  { slug: "confusion", short: "Confusion map", group: "Under the hood", title: "Which labels share a set",
    blurb: "A map of the labels that end up in the same set: the model's doubts, made visible.",
    tour: T(["map", "Row a, column b: of the sets containing a, the share that also contain b."], ["pairs", "The pairs the model confuses most."]) },
  { slug: "diff", short: "Method diff", group: "Under the hood", title: "Two rules, only where they disagree",
    blurb: "Compare two rules picture by picture: labels one adds, how many were the truth, and the pictures where they differ.",
    tour: T(["controls", "Pick rule A and rule B."], ["counts", "Labels B adds over A, and how many of them were the true label."], ["gallery", "Only the pictures whose sets differ."]) },
  { slug: "features", short: "Feature distance", group: "Under the hood", title: "Confidence against feature distance",
    blurb: "Each test picture placed by the model's confidence and by how far its features sit from the predicted class; foreign pictures in a second colour.",
    tour: T(["scatter", "Familiar pictures sit top-left: confident and close. Foreign ones drift right."]) },
  { slug: "imagenet", short: "ImageNet gallery", group: "Play", chapter: "results", title: "Hard ImageNet pictures, 1000 classes",
    blurb: "36 ImageNet pictures and the set each rule gives them.",
    tour: T(["controls", "Pick a rule, or compare all rules on each picture."]) },
  { slug: "draw", short: "Draw a digit", group: "Play", title: "Draw a digit, get an honest answer",
    blurb: "Draw on the canvas: the trained MNIST network runs in your browser and answers with a conformal set.",
    tour: T(["canvas", "Draw a digit with mouse, finger or pen. Or load a test digit."], ["answer", "Clear digits get one label; messy ones get several. Try a 4 that could be a 9."]) },
];

export const explorer = (slug: string) => EXPLORERS.find((e) => e.slug === slug);
