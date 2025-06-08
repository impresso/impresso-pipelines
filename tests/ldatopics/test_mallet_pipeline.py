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


def test_basics_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text)
    assert isinstance(result, dict)
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert isinstance(result['uid'], str)
    assert isinstance(result['ts'], str)
    assert isinstance(result['language'], str)
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)
    assert isinstance(result['min_relevance'], float)
    assert isinstance(result['topic_model_id'], str)
    assert isinstance(result['topic_model_description'], str)

def test_language_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, language='de')
    assert result['language'] == 'de'

def test_doc_name_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, doc_name='test_doc_de')
    assert result['uid'] == 'test_doc_de'

def test_min_p_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, min_relevance=0.5)
    assert result['min_relevance'] == 0.5

def test_diagnostics_lemmatization_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, diagnostics_lemmatization=True)
    assert isinstance(result, dict)
    assert 'diagnostics_lemmatization' in result.keys()
    assert isinstance(result['diagnostics_lemmatization'], list)

def test_diagnostics_topics_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, diagnostics_topics=True)
    assert isinstance(result, dict)
    assert 'diagnostics_topics' in result.keys()
    assert isinstance(result['diagnostics_topics'], dict)

def test_all_at_once_de():
    lda_pipeline = LDATopicsPipeline()
    text = """Das Leben in einem kleinen Dorf ist ruhig und wird von den Jahreszeiten bestimmt.
        Jeden Morgen grüßen sich die Bewohner, wenn sie sich in den engen Straßen begegnen,
        und der Duft von frischem Brot aus der Bäckerei erfüllt die Luft. Die Kinder gehen
        zu Fuß zur Schule, die Händler öffnen ihre Läden, und die Alten sitzen auf den Bänken,
        um über das Wetter zu plaudern. Hier kennt jeder jeden, und gegenseitige Hilfe gehört zum Alltag."""
    result = lda_pipeline(text, language='de', doc_name='test_name_de', min_relevance=0.05, diagnostics_lemmatization=True, diagnostics_topics=True)
    assert isinstance(result, dict)
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert 'diagnostics_lemmatization' in result.keys()
    assert 'diagnostics_topics' in result.keys()
    assert isinstance(result['uid'], str)
    assert isinstance(result['ts'], str)
    assert isinstance(result['language'], str)
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)
    assert isinstance(result['min_relevance'], float)
    assert isinstance(result['topic_model_id'], str)
    assert isinstance(result['topic_model_description'], str)
    assert isinstance(result['diagnostics_lemmatization'], list)
    assert isinstance(result['diagnostics_topics'], dict)
    assert result['language'] == 'de'
    assert result['uid'] == 'test_name_de'
    assert result['min_relevance'] == 0.05

def test_basics_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text)
    assert isinstance(result, dict)
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert isinstance(result['uid'], str)
    assert isinstance(result['ts'], str)
    assert isinstance(result['language'], str)
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)
    assert isinstance(result['min_relevance'], float)
    assert isinstance(result['topic_model_id'], str)
    assert isinstance(result['topic_model_description'], str)

def test_language_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, language='lb')
    assert result['language'] == 'lb'

def test_doc_name_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, doc_name='test_doc_lb')
    assert result['uid'] == 'test_doc_lb'

def test_min_p_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, min_relevance=0.5)
    assert result['min_relevance'] == 0.5

def test_diagnostics_lemmatization_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, diagnostics_lemmatization=True)
    assert isinstance(result, dict)
    assert 'diagnostics_lemmatization' in result.keys()
    assert isinstance(result['diagnostics_lemmatization'], list)

def test_diagnostics_topics_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, diagnostics_topics=True)
    assert isinstance(result, dict)
    assert 'diagnostics_topics' in result.keys()
    assert isinstance(result['diagnostics_topics'], dict)

def test_all_at_once_lb():
    lda_pipeline = LDATopicsPipeline()
    text = """D'Liewen an engem klenge Duerf ass roueg a gëtt vun de Joreszäiten bestëmmt.
        All Moien begréissen d'Awunner sech, wann se sech an den enge Stroosse begéinen,
        an de Geroch vu frëschem Brout aus der Bäckerei fëllt d'Loft. D'Kanner ginn zu Fouss an d'Schoul,
        d'Geschäftsleit maachen hir Butteker op, an d'Al sëtzen op de Bänken, fir iwwer d'Wieder ze schwätzen.
        Hei kennt jiddereen jiddereen, an Hëllef gehéiert zum Alldag."""
    result = lda_pipeline(text, language='lb', doc_name='test_name_lb', min_relevance=0.05, diagnostics_lemmatization=True, diagnostics_topics=True)
    assert isinstance(result, dict)
    assert 'uid' in result.keys()
    assert 'ts' in result.keys()
    assert 'language' in result.keys()
    assert 'topics' in result.keys()
    assert 'min_relevance' in result.keys()
    assert 'topic_model_id' in result.keys()
    assert 'topic_model_description' in result.keys()
    assert 'diagnostics_lemmatization' in result.keys()
    assert 'diagnostics_topics' in result.keys()
    assert isinstance(result['uid'], str)
    assert isinstance(result['ts'], str)
    assert isinstance(result['language'], str)
    assert isinstance(result['topics'], list)
    assert isinstance(result['topics'][0], dict)
    assert isinstance(result['min_relevance'], float)
    assert isinstance(result['topic_model_id'], str)
    assert isinstance(result['topic_model_description'], str)
    assert isinstance(result['diagnostics_lemmatization'], list)
    assert isinstance(result['diagnostics_topics'], dict)
    assert result['language'] == 'lb'
    assert result['uid'] == 'test_name_lb'
    assert result['min_relevance'] == 0.05

