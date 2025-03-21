
from impresso_pipelines.mallet.mallet_pipeline import MalletPipeline



text = "Die Katzen spielten fröhlich im Garten, während die Vögel sangen."
mallet = MalletPipeline()
output = mallet(text)
print(output)



# monolingual_lda_inferencer


# (pipelines_env_3.12) bashmak@ai:~/Desktop/GitHub_Pipelines/impresso-pipelines$ python -m impresso_pipelines.mallet.mallet_topic_inferencer     --input impresso_pipelines/mallet/output.mallet     --languages de     --de_config impresso_pipelines/mallet/tm-de-all-v2.0.config.json     --de_inferencer impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.inferencer     --de_pipe impresso_pipelines/mallet/tm-de-all-v2.0.pipe     --output output.jsonl     --output-format jsonl
# <Logger impresso_pipelines.mallet.input_reader (WARNING)>
# 2025-03-20 22:04:59,553 mallet_topic_inferencer.py:965 INFO: Script called with args: Namespace(logfile=None, level='DEBUG', input='impresso_pipelines/mallet/output.mallet', input_format='jsonl', languages=['de'], ci_ids=None, output='output.jsonl', output_format='jsonl', lemmatization_mode='v2.0-legacy', min_p=0.02, quit_if_s3_output_exists=False, s3_output_dry_run=False, s3_output_path=None, git_version=None, lingproc_run_id=None, keep_timestamp_only=False, log_file=None, quiet=False, de_config='impresso_pipelines/mallet/tm-de-all-v2.0.config.json', fr_config=None, lb_config=None, language_file=None, keep_tmp_files=False, model_dir=None, output_path_base=None, include_lid_path=False, inferencer_random_seed=42, impresso_model_id=None, de_inferencer='impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.inferencer', de_pipe='impresso_pipelines/mallet/tm-de-all-v2.0.pipe', de_lemmatization=None, fr_inferencer=None, fr_pipe=None, fr_lemmatization=None, lb_inferencer=None, lb_pipe=None, lb_lemmatization=None, de_model_id=None, fr_model_id=None, lb_model_id=None, de_topic_count=None, fr_topic_count=None, lb_topic_count=None)
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:967 INFO: Setting up MalletTopicInferencer
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:980 INFO: Config's impresso_pipelines/mallet/tm-de-all-v2.0.config.json topic count for language de used: 100
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:1046 INFO: Performing monolingual topic inference for the following languages: ['de']
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:1050 INFO: MalletTopicInferencer setup finished.
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:1051 INFO: MalletTopicInferencer Class Arguments: Namespace(logfile=None, level='DEBUG', input='impresso_pipelines/mallet/output.mallet', input_format='jsonl', languages=['de'], ci_ids=None, output='output.jsonl', output_format='jsonl', lemmatization_mode='v2.0-legacy', min_p=0.02, quit_if_s3_output_exists=False, s3_output_dry_run=False, s3_output_path=None, git_version=None, lingproc_run_id=None, keep_timestamp_only=False, log_file=None, quiet=False, de_config='impresso_pipelines/mallet/tm-de-all-v2.0.config.json', fr_config=None, lb_config=None, language_file=None, keep_tmp_files=False, model_dir=None, output_path_base=None, include_lid_path=False, inferencer_random_seed=42, impresso_model_id=None, de_inferencer='impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.inferencer', de_pipe='impresso_pipelines/mallet/tm-de-all-v2.0.pipe', de_lemmatization=None, fr_inferencer=None, fr_pipe=None, fr_lemmatization=None, lb_inferencer=None, lb_pipe=None, lb_lemmatization=None, de_model_id='tm-de-all-v2.0', fr_model_id=None, lb_model_id=None, de_topic_count=100, fr_topic_count=None, lb_topic_count=None)
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:209 WARNING: JVM already running.
# 2025-03-20 22:04:59,554 mallet_topic_inferencer.py:286 INFO: Loaded configuration for language: de : impresso_pipelines/mallet/tm-de-all-v2.0.config.json : {'uposFilter': ['NOUN', 'PROPN'], 'topic_count': 100, 'language': 'de', 'model_id': 'tm-de-all-v2.0', 'lowercase_token': False, 'min_lemmas': 8}
# 2025-03-20 22:04:59,556 mallet_topic_inferencer.py:339 INFO: Lemmatization file for language: de not provided by arguments. Skipping.
# 2025-03-20 22:04:59,556 mallet_topic_inferencer.py:368 INFO: Initializing Mallet2TopicAssignment converter for de
# 2025-03-20 22:04:59,557 mallet2topic_assignment_jsonl.py:528 INFO: Mallet2TopicAssignment Options: Namespace(logfile=None, lang='de', topic_model='tm-de-all-v2.0', numeric_topic_ids=False, min_p=0.02, input_format_type='matrix', topic_count=100, output='<generator>', impresso_model_id=None, lingproc_run_id=None, git_version=None, no_jsonschema_validation=False, level='INFO', INPUT_FILES=[])
# 2025-03-20 22:04:59,639 mallet_topic_inferencer.py:455 INFO: Processing input file: impresso_pipelines/mallet/output.mallet
# 2025-03-20 22:04:59,707 language_inferencer.py:149 INFO: Calling mallet InferTopics: ['--input', 'impresso_pipelines/mallet/output.mallet', '--inferencer', 'impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.inferencer', '--output-doc-topics', 'impresso_pipelines/mallet/output.mallet.doctopics', '--random-seed', '42']
# 2025-03-20 22:04:59,989 mallet2topic_assignment_jsonl.py:528 INFO: Mallet2TopicAssignment Options: Namespace(logfile=None, lang='de', topic_model='tm-de-all-v2.0', numeric_topic_ids=False, min_p=0.02, input_format_type='matrix', topic_count=100, output='<generator>', impresso_model_id=None, lingproc_run_id=None, git_version=None, no_jsonschema_validation=False, level='INFO', INPUT_FILES=['impresso_pipelines/mallet/output.mallet.doctopics'])
# 2025-03-20 22:05:00,073 mallet2topic_assignment_jsonl.py:311 INFO: DUPLICATE-COUNT: 0
# 2025-03-20 22:05:00,073 mallet_topic_inferencer.py:223 INFO: STATS: content_items: 2