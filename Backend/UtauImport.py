from Backend.DataHandler.UtauSample import UtauSample

def fetchSamples(filename, properties):
    alias = properties[0]
    offset = properties[1]
    fixed = properties[2]
    blank = properties[3]
    preuttr = properties[4]
    overlap = properties[5]
    
    return []