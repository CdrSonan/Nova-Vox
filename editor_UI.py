import tkinter.filedialog

import editor_pipeline

filepath = tkinter.filedialog.askopenfilename(filetypes = ((".nvvb Voicebanks", ".nvvb"), ("all_files", "*")))
if filepath != "":
    vb = editor_pipeline.Voicebank(filepath)
    filepath = tkinter.filedialog.askopenfilename(filetypes = (("text files", ".txt"), ("all_files", "*")))
    if filepath != "":
        with open(filepath, 'r') as f:
            exec(f.read())