# Streamlit app cold-start measurement

Issue #36 requires measuring the first load of the app separately from repeated
queries. Start the app with a clean process:

```powershell
python -m streamlit run src/app/app.py
```

Record the time from opening the app until the question box is usable. Then
record the time for the first submitted question. Repeat the same question
without changing its filters; the second run should use `answer_cached` and be
materially faster.

The app displays the completed-query duration below the answer. The first
query includes loading the indexes, embedding model and (when selected) the
cross-encoder. Streamlit's `@st.cache_resource` keeps those resources alive for
the process, while `@st.cache_data` reuses the answer for an identical question,
filter tuple and retrieval method.

Record results in the project report using this format:

| Run | Retrieval method | Cold/warm | Time |
|---|---|---|---|
| 1 | Hybrid + rerank | Cold | __ s |
| 2 | Hybrid + rerank | Warm, same question | __ s |
