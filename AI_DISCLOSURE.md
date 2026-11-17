# AI use disclosure

## Tools used

- **Claude Code** (Anthropic) with the models Claude Fable 5.2 and Claude Opus 5.5.
- **ChatGPT** (OpenAI) with model GPT 5.6 Luna.

## What they were used for

- **Website (`site/`).** Claude Code wrote the HTML, CSS and JavaScript: `site/index.html` and
  `site/assets/core.js`, `story.js`, `model.js`, `results.js` and `styles.css`. It also wrote the
  interactive figures and animations: the prerocessing stepper, the split demo, the filter explorer, the
  rotation playground, the confusion-matrix explorer, the bootstrap demo and the real-fab slider.
- **Data export for the website (`scripts/11_site_data.py`, `python main.py site-data`).** Claude Code wrote
  the script that reads `experiments/` and `splits/`, writes `site/data/*.js` and
  `experiments/site_stats.json`, and fills the numbers into `site/index.html`.
- **Publishing.** The GitHub Pages workflow (`.github/workflows/pages.yml`) and the "Website" section of
  `README.md`.
- **Photos.** Claude Code found the four reference photos on Wikimedia Commons, checked their licences, and
  cropped them. Each photo is credited on the page with its author and licence.
- **Code.** Claude Code wrote the code for the mathematical ML concepts we wanted to implement.
- **Brainstorming and Resources.** ChatGPT was used for brainstorming and finding resources for the project during the initial phases of the project.

## What we did ourselves

- **Design decisions:** We defined the main research question, split the data by wafer lot to avoid data leakage, and chose macro-F1 as our primary evaluation metric. We then fixed our final pipeline (k-means with K = 32, four rotations, ring pooling and isolated-die cleanup). Finally, we planned experiments covering ablations, rotation and shift robustness, comparison with a standard CNN, and replaying the model on real fabrication data.
- **Experiments and results:** We reviewed the results saved by the scripts in `experiments/` and took decisions on how to proceed based on the results. 
- **The website's content plan:** We prepared the section order, the six-part layout for each section and all the other decisions for the design.

## Numbers

Every number in this repo and on the website comes from a run logged in `experiments/log.csv` or the JSON/CSV
files next to it, not from an AI tool. The website's numbers are copied from those files by
`scripts/11_site_data.py`.
