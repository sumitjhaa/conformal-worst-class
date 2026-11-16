// Copies the exported results (site/data, including mnist_net.json from scripts/export_mnist_net.py), the GIFs (site/gifs) and the paper PDFs into public/.
// Run automatically before `yarn dev` and `yarn build`; regenerate site/data with `python scripts/export_site.py`.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const root = join(web, "..");
const pub = join(web, "public");
const need = join(root, "site", "data", "results.json");
if (!existsSync(need)) {
  console.error(`missing ${need}: run \`python scripts/export_site.py\` in the project root first`);
  process.exit(1);
}
cpSync(join(root, "site", "data"), join(pub, "data"), { recursive: true });
// the old standalone explorer (site/index.html + app.js) is replaced by the Next routes under src/app/explorer
if (existsSync(join(root, "site", "gifs"))) cpSync(join(root, "site", "gifs"), join(pub, "gifs"), { recursive: true }); // python scripts/make_gifs.py
mkdirSync(join(pub, "papers"), { recursive: true });
for (const f of ["26210180_Sumit_Jha.pdf", "conformal_worst_class.pdf"]) {
  const p = join(root, "paper", f);
  if (existsSync(p)) cpSync(p, join(pub, "papers", f));
}
console.log("synced data and papers into public/");
