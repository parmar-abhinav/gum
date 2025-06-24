TRANSCRIPTION_PROMPT = """Transcribe in markdown ALL the content from the screenshots of the user's screen.

NEVER SUMMARIZE ANYTHING. You must transcribe everything EXACTLY, word for word, but don't repeat yourself.

ALWAYS include all the application names, file paths, and website URLs in your transcript.

Create a FINAL structured markdown transcription."""

SUMMARY_PROMPT = """Provide a detailed description of the actions occuring across the provided images. 

Include as much relevant detail as possible, but remain concise.

Generate a handful of bullet points and reference *specific* actions the user is taking."""