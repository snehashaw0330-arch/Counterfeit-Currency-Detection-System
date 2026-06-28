APP SCREENSHOTS FOR THE PROJECT REPORT
======================================

These are the real screenshots of the running application. scripts/
build_report_doc.py embeds them into "APPENDIX A: SCREENSHOTS" of
docs/Project_Report.docx, in this order:

  01_home_upload.png         -> Home / upload screen (English/Hindi toggle)
  02_verdict_genuine.png     -> Verdict: Likely Genuine Rs.100 (REAL) + overlay
  03_what_we_checked.png     -> "What we checked" plain-language panel
  04_explain_ai.png          -> "Explain with AI" explanation + by-hand checks
  05_ml_comparison.png       -> Technical details: ML technique comparison
  06_forensic_analysis.png   -> Forensic analysis (proportion + checks grid)
  07_serial_typography.png   -> RBI serial-typography check (ascending digits)
  08_pattern_and_chatbot.png -> Security Pattern Studio + Help chatbot

The report loads these from FILES (not pasted into Word by hand), so
regenerating the document never loses them — just keep the files here.

To replace any image, overwrite the file of the same name (PNG or JPG) and
rebuild:

  venv\Scripts\python.exe scripts\build_report_doc.py

A missing file simply shows a labelled placeholder naming the file to drop in,
so a partial set still builds cleanly. To split the pattern studio and the
chatbot into two separate figures, save 08_security_pattern.png and
09_help_chatbot.png and tell the build script to use them.
