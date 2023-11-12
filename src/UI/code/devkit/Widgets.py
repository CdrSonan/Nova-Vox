from tkinter import Frame as TkFrame, Label as TkLabel, Button as TkButton, LabelFrame as TkLabelFrame, Listbox as TkListbox, Scale as TkScale, Entry as TkEntry, Spinbox as TkSpinbox, Checkbutton as TkCheckbutton, Radiobutton as TkRadiobutton

class Frame(TkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, background="#333333")
        
class Label(TkLabel):
    def __init__(self, master) -> None:
        super().__init__(master, activebackground="#333333", activeforeground="#FFF", background="#333333", foreground="#FFF")
        
class Button(TkButton):
    def __init__(self, master) -> None:
        super().__init__(master, activebackground="#2d0033", activeforeground="#FFF", background="#333333", borderwidth=3, disabledforeground="#666", foreground="#FFF", overrelief="ridge", relief="solid")
        
class SlimButton(TkButton):
    def __init__(self, master) -> None:
        super().__init__(master, activebackground="#2d0033", activeforeground="#FFF", background="#333333", borderwidth=1, disabledforeground="#666", foreground="#FFF", overrelief="ridge", relief="solid")

class LabelFrame(TkLabelFrame):
    def __init__(self, master, text) -> None:
        super().__init__(master, background="#404040", borderwidth=3, foreground="#FFF", relief="solid", text=text)

class Listbox(TkListbox):
    def __init__(self, master) -> None:
        super().__init__(master, background="#23202c", disabledforeground="#666", foreground="#FFF", selectbackground="#2d0033", selectforeground="#FFF", borderwidth=0, relief="solid")

class Entry(TkEntry):
    def __init__(self, master) -> None:
        super().__init__(master, background="#333333", borderwidth=1, disabledbackground="#333333", disabledforeground="#666", foreground="#FFF", insertbackground="#FFF", relief="solid", selectbackground="#2d0033", selectforeground="#FFF")

class Spinbox(TkSpinbox):
    def __init__(self, master, from_, to, increment=1.) -> None:
        super().__init__(master, activebackground="#2d0033", background="#333333", borderwidth=1, buttonbackground="#333333", disabledbackground="#333333", disabledforeground="#666", foreground="#FFF", insertbackground="#FFF", relief="solid", selectbackground="#2d0033", from_=from_, to=to, increment=increment)

class Checkbutton(TkCheckbutton):
    def __init__(self, master) -> None:
        super().__init__(master, activebackground="#333333", activeforeground="#2d0033", background="#333333", disabledforeground="#666")

class Radiobutton(TkRadiobutton):
    def __init__(self, master, text, value, variable, command) -> None:
        super().__init__(master, activebackground="#2d0033", activeforeground="#FFF", background="#333333", disabledforeground="#666", foreground="#FFF", selectcolor="#000", text=text, value=value, variable=variable, command=command)

class Scale(TkScale):
    def __init__(self, master, from_, to, orient, length, command) -> None:
        super().__init__(master, activebackground="#2d0033", background="#333333", borderwidth=0, foreground="#FFF", relief="solid", sliderrelief="solid", troughcolor="#23202c", from_=from_, to=to, orient=orient, length=length, command=command)
