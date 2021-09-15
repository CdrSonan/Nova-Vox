filepath = tkinter.filedialog.askopenfilename(filetypes = ((".nvvb Voicebanks", ".nvvb"), ("all_files", "*")))
if filepath != "":
    vb = Voicebank(filepath)
    filepath = tkinter.filedialog.askopenfilename(filetypes = (("text files", ".txt"), ("all_files", "*")))
    if filepath != "":
        with open(filepath, 'r') as f:
            exec(f.read())