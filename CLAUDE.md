# JaMoNA — Jimena and Andrea's Model Network Analysis

- This is homework of a course, so do not over-engineer. Still, code must be prolific, clean, and not excessively simplified.

- Access to both Colab (with T4 gpu) and local computer.

- Use all datasets.

- Do not commit nor push yourself to git repo.

## Plan tracking

- `PLAN.md` is the source of truth for the homework plan and its progress. Read it before starting any work.
- Status markers: `[ ]` not started, `[~]` in progress, `[x]` done, `[-]` dropped (with reason).
- Whenever you start or finish a plan item, update its marker in `PLAN.md` in the same session; add a short
  `> note:` under the item for anything the next person must know (paths, deviations, gotchas).
- Append one line per working session to the **Progress log** table at the end of `PLAN.md`.
- If a key decision changes, edit the **Key decisions** table and log the reason.

## Reading the paper

- The PDF in `paper/` is >20 MB, so the Read tool can't open it directly. Use PyMuPDF from the root venv
  (`.venv/Scripts/python.exe`, `import pymupdf`) to extract text or render pages to PNG.

- Use the `.venv/` to install python libraries.

## Data Visualization

- Avoid textual explanations on plots. If a plot must be explained, opt for an info icon rather than presenting the explanation on the plot. 

- Use visualizations in `style-references/` for aesthetic/style references. 