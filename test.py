
from impresso_pipelines.ldatopics.mallet_pipeline import MalletPipeline



text = "Die Katzen spielten fröhlich im Garten, während die Vögel sangen."
text_fr = "Les chats jouaient joyeusement dans le jardin pendant que les oiseaux chantaient."
text_lb = "Katzen hunn am Gaart gespillt, während Villercher gesongen hunn."
mallet = MalletPipeline()
print(mallet(text, doc_name = "my_doc"))
# print empty line
print()
print()
print()
# print(mallet(text_fr))
print(mallet(text_lb))

print(mallet(text_lb, doc_name = "another_doc"))


print(mallet(text_lb))