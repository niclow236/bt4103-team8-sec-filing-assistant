# Streamlit app cold-start measurement

Issue #36 asks for the app's first load to be measured apart from the queries
after it. Start the app in a new process:

```powershell
streamlit run src/app/main.py
```

## What the app keeps

Streamlit runs the page's script again on every widget change, so the app
keeps three things outside it, for as long as the process runs:

- The indexes and the embedding model. `load_stack` builds the configuration
  picked in the sidebar once and `st.cache_resource` keeps it. Every
  configuration shares one BM25 index, one dense index and one embedding model
  (`_indexes`), so picking another row does not load them again.
- The provider and model from `.env`, which are read when a configuration is
  built.
- The answers. A question asked again with the same filters under the same
  configuration is given the answer it was given before, with no search and no
  model call, in the same browser session or another (`src/app/state.py`). A
  failure is not kept, so a question that met a busy provider is asked again,
  and neither is an answer the model left malformed or cut short. Two sessions
  that ask the same question at the same moment get one model call between
  them: the second waits for the first's answer. Each configuration keeps its
  128 most recently asked answers (`ANSWERS_KEPT`).

A changed `.env`, a rebuilt index or a wish for a fresh answer to a remembered
question needs the app restarted.

The app shows how long an answer took above it ("Query completed in 0.34s")
and keeps the line with the answer. The first Ask of a process includes
loading the indexes and the embedding model.

## Measured

On one laptop (Intel i5-1135G7, 16 GB RAM, no GPU use), on 3 Oct 2026, on the
full 15-company index of 28,289 passages. Each step ran the app's entry point
(`src/app/app.py` at the time, now `src/app/main.py`) in a new Python process
through Streamlit's `AppTest`, which runs the script as a browser session
does, and each time is the app's own "Query completed in" line. The first Ask
was measured twice.

The questions were ones the facts store answers ("What was Apple's total
revenue in FY2024?"), so no model was asked. The times are the indexes, the
embedding model, the search and the memory. A question the store does not
answer adds the model's time: seconds on Mistral's API, minutes on the local
Ollama model.

| Step | Configuration | Time |
|---|---|---|
| `streamlit run` until the server answers | any | 1.2 s |
| The page's first render in a new process | any | 1.9 s |
| First Ask in a new process | C4, hybrid | 21.3 s, 21.4 s |
| A new question once the process is warm | C4 | 0.3 s |
| The same question again, in the same or another browser session | C4 | 0.1 s |
| First Ask in a new process, up to the model call | C1, BM25 | 6.4 s |
| Another configuration's first Ask in the warm process, up to the model call | C3, then C1 | 2.3 s, 2.5 s |

A repeated question takes a tenth of a second and not none because the answer
is checked again before it is shown. Only the search and the model call are
skipped.

The last two rows stop at the model call. C1 to C3 were measured without the
metadata filter, so they pass the facts route no company or year and the same
question goes to a model. The provider was pointed at a closed port, and about
2 s of each of those times is the refused connection: a second question under
C1 took 2.2 s the same way. So BM25 alone loads in about 4 s, and switching
from C4 to C3 or C1 loads nothing.

The index files had been read earlier that day, so they were in the operating
system's file cache. Right after a reboot they are not, and the first Ask
takes minutes: 239 s in an earlier measurement on this laptop, which also
loaded a cross-encoder the app no longer has and asked a hosted model.

Hardware, index size, model and provider all move these figures.
