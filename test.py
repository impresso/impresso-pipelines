
from impresso_pipelines.mallet.mallet_pipeline import MalletPipeline



text = "Die Katzen spielten fröhlich im Garten, während die Vögel sangen."
text_fr = "Les chats jouaient joyeusement dans le jardin pendant que les oiseaux chantaient."
text_lb = "D'Katzen hunn am Gaart gespillt, während d'Villercher gesongen hunn."
mallet = MalletPipeline()
output = mallet(text_lb)
print(output)


