# Streamlit app cold-start measurement

Issue #36 requires measuring the first load of the app separately from repeated
queries. Start the app with a clean process:

```powershell
python -m streamlit run src/app/app.py
```

The app displays the completed-query duration below the answer. The first
query includes loading the indexes, embedding model and (when selected) the
cross-encoder. Streamlit's `@st.cache_resource` keeps those resources alive for
the process, while `@st.cache_data` reuses the answer for an identical question,
filter tuple and retrieval method.

## Recorded baseline

The team's recorded laptop measurement was approximately **41 seconds for a
cold start**, including index and model loading. Once the process was warm, the
first search took approximately **13-15 seconds**. Repeated identical questions
use the data cache and do not call `answer_question` again. These figures are a
baseline rather than a universal guarantee: hardware, local index size, model,
and provider affect the result.

| Run | Retrieval method | Cold/warm | Time |
|---|---|---|---|
| 1 | Hybrid + rerank | Cold | ~41 s |
| 2 | Hybrid + rerank | Warm, first search | ~13-15 s |
| 3 | Hybrid + rerank | Cached repeated question | No second answer-generation call |
