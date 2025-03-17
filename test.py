
from impresso_pipelines.mallet.mallet_pipeline import MalletPipeline



text = "Die Katzen spielten fröhlich im Garten, während die Vögel sangen."
mallet = MalletPipeline()
output = mallet(text)
print(output)