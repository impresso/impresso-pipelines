
from impresso_pipelines.mallet.mallet_pipeline import MalletPipeline



text = "Die Katzen spielten fröhlich im Garten, während die Vögel sangen."
text_fr = "Les chats jouaient joyeusement dans le jardin pendant que les oiseaux chantaient."
text_lb = "Katzen hunn am Gaart gespillt, während Villercher gesongen hunn."
mallet = MalletPipeline()
print(mallet(text))
# print empty line
print()
print()
print()
# print(mallet(text_fr))
print(mallet(text_lb))


