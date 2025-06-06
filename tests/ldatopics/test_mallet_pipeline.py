import pytest
from impresso_pipelines.ldatopics.mallet_pipeline import LDATopicsPipeline

def test_basics():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text)

    # assert that the pipeline returns a list
    

    # assert that first and only element of the list is a dictionary
    assert isinstance(result, dict)
    
    # assert that dictionary has keys 'uid', 'ts', 'language', 'topics', 'min_relevance', 'topic_model_id', 'topic_model_description'
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()

    # assert that 'uid' is a string
    assert isinstance(result['uid'], str)

    # assert that 'ts' is a string
    assert isinstance(result['ts'], str)

    # assert that 'language' is a string
    assert isinstance(result['language'], str)
    
    # assert that 'topics' is a list of dictionaries
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)

    # assert that 'min_relevance' is a float
    assert isinstance(result['min_relevance'], float)

    # assert that 'topic_model_id' is a string
    assert isinstance(result['topic_model_id'], str)

    # assert that 'topic_model_description' is a string
    assert isinstance(result['topic_model_description'], str)



def test_language():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, language='lb')

    # assert that language is 'lb'
    assert result['language'] == 'lb'

    

def test_doc_name():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, doc_name='test_doc')

    # assert that doc_name is 'test_doc'
    assert result['uid'] == 'test_doc'


def test_min_p():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, min_relevance=0.5)

    # assert that min_relevance is 0.5
    assert result['min_relevance'] == 0.5


def test_diagnostics_lemmatization():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, diagnostics_lemmatization=True)

    # assert that the pipeline returns a list
    

    # assert that first and only element of the list is a dictionary
    assert isinstance(result, dict)
    
    # assert that dictionary has keys 'uid', 'ts', 'language', 'topics', 'min_relevance', 'topic_model_id', 'topic_model_description', 'diagnostics_lemmatization'
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert 'diagnostics_lemmatization' in result.keys()

    # assert that 'uid' is a string
    assert isinstance(result['uid'], str)

    # assert that 'ts' is a string
    assert isinstance(result['ts'], str)

    # assert that 'language' is a string
    assert isinstance(result['language'], str)
    
    # assert that 'topics' is a list of dictionaries
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)

    # assert that 'min_relevance' is a float
    assert isinstance(result['min_relevance'], float)

    # assert that 'topic_model_id' is a string
    assert isinstance(result['topic_model_id'], str)

    # assert that 'topic_model_description' is a string
    assert isinstance(result['topic_model_description'], str)
    
    # assert that 'diagnostics_lemmatization' is a list
    assert isinstance(result['diagnostics_lemmatization'], list)


def test_diagnostics_topics():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, diagnostics_topics=True)

    # assert that the pipeline returns a list
    

    # assert that first and only element of the list is a dictionary
    assert isinstance(result, dict)
    
    # assert that dictionary has keys 'uid', 'ts', 'language', 'topics', 'min_relevance', 'topic_model_id', 'topic_model_description', 'diagnostics_topics'
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert 'diagnostics_topics' in result.keys()

    # assert that 'uid' is a string
    assert isinstance(result['uid'], str)

    # assert that 'ts' is a string
    assert isinstance(result['ts'], str)

    # assert that 'language' is a string
    assert isinstance(result['language'], str)
    
    # assert that 'topics' is a list of dictionaries
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)

    # assert that 'min_relevance' is a float
    assert isinstance(result['min_relevance'], float)

    # assert that 'topic_model_id' is a string
    assert isinstance(result['topic_model_id'], str)

    # assert that 'topic_model_description' is a string
    assert isinstance(result['topic_model_description'], str)

    # assert that 'diagnostics_topics' is a dict
    assert isinstance(result['diagnostics_topics'], dict)


def test_all_at_once():
    lda_pipeline = LDATopicsPipeline()
    text = """La vie dans un petit village est paisible et rythmée par les saisons.
        Chaque matin, les habitants se saluent en se croisant dans les rues étroites,
        et l’odeur du pain frais s’échappant de la boulangerie emplit l’air. Les enfants
        vont à l’école à pied, les commerçants ouvrent leurs boutiques, et les anciens
        s’assoient sur les bancs pour discuter de la pluie et du beau temps. Ici, tout
        le monde se connaît, et l’entraide fait partie du quotidien."""
    
    result = lda_pipeline(text, language='lb', doc_name='test_name', min_relevance=0.05, diagnostics_lemmatization=True, diagnostics_topics=True)

    # assert that the pipeline returns a list
    

    # assert that first and only element of the list is a dictionary
    assert isinstance(result, dict)
    
    # assert that dictionary has keys 'uid', 'ts', 'language', 'topics', 'min_relevance', 'topic_model_id', 'topic_model_description', 'diagnostics_lemmatization', 'diagnostics_topics'
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert 'diagnostics_lemmatization' in result.keys()
    assert 'diagnostics_topics' in result.keys()

    # assert that 'uid' is a string
    assert isinstance(result['uid'], str)

    # assert that 'ts' is a string
    assert isinstance(result['ts'], str)

    # assert that 'language' is a string
    assert isinstance(result['language'], str)
    
    # assert that 'topics' is a list of dictionaries
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)

    # assert that 'min_relevance' is a float
    assert isinstance(result['min_relevance'], float)

    # assert that 'topic_model_id'
    assert isinstance(result['topic_model_id'], str)

    # assert that 'topic_model_description' is a string
    assert isinstance(result['topic_model_description'], str)

    # assert that 'diagnostics_lemmatization' is a list
    assert isinstance(result['diagnostics_lemmatization'], list)

    # assert that 'diagnostics_topics' is a dict
    assert isinstance(result['diagnostics_topics'], dict)

    # assert that 'language' is 'lb'
    assert result['language'] == 'lb'

    # assert that 'uid' is 'test_name'
    assert result['uid'] == 'test_name'

    # assert that 'min_relevance' is 0.05
    assert result['min_relevance'] == 0.05

