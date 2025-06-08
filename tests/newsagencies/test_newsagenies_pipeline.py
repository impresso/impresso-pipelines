import pytest_lazyfixture
from impresso_pipelines.newsagencies.newsagencies_pipeline import NewsAgenciesPipeline



def test_basics():
    news_pipeline = NewsAgenciesPipeline()
    text = """
        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».



        Wie das Wolffs Telegraphisches Bureau (Wolff) gestern in einer abendlichen Meldung verlautbarte, trat der Haushaltsausschuss „unter beträchtlichem Andrang der Presse“ zusammen. Eine Fernschreibnotiz der Telegraphen‑Union (Telunion) fügte hinzu, man erwarte „lebhafte Zwischenrufe von der Zentrumspartei“. Gemäß Deutsches Nachrichtenbüro (DNB) erinnern die Vorgänge an die Zolltarif‑Debatten des Jahres 1879.
        Der Deutsche Depeschendienst / dapd (DDP‑DAPD) berichtet von letzter Minute‑Verhandlungen, während die Deutsche Presse‑Agentur (dpa) meldet, der Reichskanzler halte sich „zu weitergehenden Stellungnahmen bedeckt“. Aus Wien telegraphiert die Österreichische Presseagentur (APA), man hege „vorsichtigen Optimismus“; gleichzeitig warnt die Schweizerische Depeschenagentur (ATS/SDA) vor juristischen Fallstricken im Alpenraum. Die Schweizer Mittelpresse (SPK‑SMP) lässt verlauten, mehrere Kantone pochten auf Kompensationen. Nach einer Drahtnachricht der tschechischen ČTK wolle Prag demnächst ein Gutachten veröffentlichen.



        By cable to The Times of London, Reuters states that the conference hall fell silent when the chairman produced a memorandum drafted, it is said, by experts of the old Stefani bureau. Across the Atlantic, the Associated Press (AP) wires that a „spirit of compromise” pervades the corridors, though private circulars hint at lingering scepticism.
        From New York, United    Press International (UPI/UP) recalls that Domei chroniclers followed the Tokyo tariff talks of 1934 „with equal fervour“. Market sheets collated by Extel record a brief rally in overseas securities; yet commentators consulted by Europapress counsel prudence for the Latin exchanges. Warsaw-based PAP intimates that Poland will vote „in concert with Budapest and Prague“. Stockholm’s Tidningarnas Telegrambyrå (TT), meanwhile, relays Nordic caution, whereas Belgrade’s TANJUG cautions against „any settlement that might hamper Balkan exports“.
        Late in the evening, TASS issues a bulletin insisting that existing fuel conventions be honoured; a companion wire from Interfax suggests the Kremlin regards the matter as „no less vital than the grain question of 1917“.


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».

        Wie das Wolffs Telegraphisches Bureau (Wolff) gestern in einer abendlichen Meldung verlautbarte, trat der Haushaltsausschuss „unter beträchtlichem Andrang der Presse“ zusammen. Eine Fernschreibnotiz der Telegraphen‑Union (Telunion) fügte hinzu, man erwarte „lebhafte Zwischenrufe von der Zentrumspartei“. Gemäß Deutsches Nachrichtenbüro (DNB) erinnern die Vorgänge an die Zolltarif‑Debatten des Jahres 1879.
        Der Deutsche Depeschendienst / dapd (DDP‑DAPD) berichtet von letzter Minute‑Verhandlungen, während die Deutsche Presse‑Agentur (dpa) meldet, der Reichskanzler halte sich „zu weitergehenden Stellungnahmen bedeckt“. Aus Wien telegraphiert die Österreichische Presseagentur (APA), man hege „vorsichtigen Optimismus“; gleichzeitig warnt die Schweizerische Depeschenagentur (ATS/SDA) vor juristischen Fallstricken im Alpenraum. Die Schweizer Mittelpresse (SPK‑SMP) lässt verlauten, mehrere Kantone pochten auf Kompensationen. Nach einer Drahtnachricht der tschechischen ČTK wolle Prag demnächst ein Gutachten veröffentlichen.
    """ 

    result = news_pipeline(text)

    # check that it returns a key 'agencies' and value which is a list
    assert 'agencies' in result
    assert isinstance(result['agencies'], list)
    # check that the list has at least one item
    assert len(result['agencies']) > 0
    # check that the first item in the list is a dictionary
    assert isinstance(result['agencies'][0], dict)
    # check that the first item in the list has keys 'uid' with a string, relevance with a float, and wikidata_link with a string
    assert 'uid' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['uid'], str)
    assert 'relevance' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['relevance'], float)
    assert 'wikidata_link' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['wikidata_link'], str)


def test_diagnostics():
    news_pipeline = NewsAgenciesPipeline()
    text = """
        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».



        Wie das Wolffs Telegraphisches Bureau (Wolff) gestern in einer abendlichen Meldung verlautbarte, trat der Haushaltsausschuss „unter beträchtlichem Andrang der Presse“ zusammen. Eine Fernschreibnotiz der Telegraphen‑Union (Telunion) fügte hinzu, man erwarte „lebhafte Zwischenrufe von der Zentrumspartei“. Gemäß Deutsches Nachrichtenbüro (DNB) erinnern die Vorgänge an die Zolltarif‑Debatten des Jahres 1879.
        Der Deutsche Depeschendienst / dapd (DDP‑DAPD) berichtet von letzter Minute‑Verhandlungen, während die Deutsche Presse‑Agentur (dpa) meldet, der Reichskanzler halte sich „zu weitergehenden Stellungnahmen bedeckt“. Aus Wien telegraphiert die Österreichische Presseagentur (APA), man hege „vorsichtigen Optimismus“; gleichzeitig warnt die Schweizerische Depeschenagentur (ATS/SDA) vor juristischen Fallstricken im Alpenraum. Die Schweizer Mittelpresse (SPK‑SMP) lässt verlauten, mehrere Kantone pochten auf Kompensationen. Nach einer Drahtnachricht der tschechischen ČTK wolle Prag demnächst ein Gutachten veröffentlichen.



        By cable to The Times of London, Reuters states that the conference hall fell silent when the chairman produced a memorandum drafted, it is said, by experts of the old Stefani bureau. Across the Atlantic, the Associated Press (AP) wires that a „spirit of compromise” pervades the corridors, though private circulars hint at lingering scepticism.
        From New York, United    Press International (UPI/UP) recalls that Domei chroniclers followed the Tokyo tariff talks of 1934 „with equal fervour“. Market sheets collated by Extel record a brief rally in overseas securities; yet commentators consulted by Europapress counsel prudence for the Latin exchanges. Warsaw-based PAP intimates that Poland will vote „in concert with Budapest and Prague“. Stockholm’s Tidningarnas Telegrambyrå (TT), meanwhile, relays Nordic caution, whereas Belgrade’s TANJUG cautions against „any settlement that might hamper Balkan exports“.
        Late in the evening, TASS issues a bulletin insisting that existing fuel conventions be honoured; a companion wire from Interfax suggests the Kremlin regards the matter as „no less vital than the grain question of 1917“.


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence        Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».


        Selon une dépêche matinale de l’Agence Havas, datée du 14 mai 1898, la Chambre s’est réunie dans une atmosphère « pleine de gravité ». Par câble, l’Agence France‑Presse (AFP) précise que les bancs de la gauche ont bruyamment salué l’allocution ministérielle. D’Amsterdam, l’Algemeen Nederlands Persbureau (ANP) fait savoir que La Haye observe l’affaire « avec le plus grand intérêt », tandis que l’Agenzia Nazionale Stampa Associata (ANSA) signale de Rome qu’un amendement sera déposé « au nom de la défense des manufactures locales ».
        À Berne, ATS‑SDA expédie un avis où l’on lit que le Conseil fédéral « n’entend point se départir de la doctrine de neutralité »; Belga télégraphie que Bruxelles redoute des rétorsions douanières. De Sofia, BTA rappelle qu’en 1904 déjà, des économistes bulgares prévoyaient des secousses monétaires similaires. Dans les cafés de Saint‑Pétersbourg, rapporte Interfax, on discute du projet « comme on eût commenté la réforme de 1861 ».

        Wie das Wolffs Telegraphisches Bureau (Wolff) gestern in einer abendlichen Meldung verlautbarte, trat der Haushaltsausschuss „unter beträchtlichem Andrang der Presse“ zusammen. Eine Fernschreibnotiz der Telegraphen‑Union (Telunion) fügte hinzu, man erwarte „lebhafte Zwischenrufe von der Zentrumspartei“. Gemäß Deutsches Nachrichtenbüro (DNB) erinnern die Vorgänge an die Zolltarif‑Debatten des Jahres 1879.
        Der Deutsche Depeschendienst / dapd (DDP‑DAPD) berichtet von letzter Minute‑Verhandlungen, während die Deutsche Presse‑Agentur (dpa) meldet, der Reichskanzler halte sich „zu weitergehenden Stellungnahmen bedeckt“. Aus Wien telegraphiert die Österreichische Presseagentur (APA), man hege „vorsichtigen Optimismus“; gleichzeitig warnt die Schweizerische Depeschenagentur (ATS/SDA) vor juristischen Fallstricken im Alpenraum. Die Schweizer Mittelpresse (SPK‑SMP) lässt verlauten, mehrere Kantone pochten auf Kompensationen. Nach einer Drahtnachricht der tschechischen ČTK wolle Prag demnächst ein Gutachten veröffentlichen.
    """ 

    result = news_pipeline(text)

    # check that it returns a key 'agencies' and value which is a list
    assert 'agencies' in result
    assert isinstance(result['agencies'], list)
    # check that the list has at least one item
    assert len(result['agencies']) > 0
    # check that the first item in the list is a dictionary
    assert isinstance(result['agencies'][0], dict)
    # check that the first item in the list has keys 'uid' with a string, relevance with a float, and wikidata_link with a string
    assert 'uid' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['uid'], str)
    assert 'relevance' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['relevance'], float)
    assert 'wikidata_link' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['wikidata_link'], str)

    # check that it also has key surface with a string, start with int, stop with int
    assert 'surface' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['surface'], str)
    assert 'start' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['start'], int)
    assert 'stop' in result['agencies'][0]
    assert isinstance(result['agencies'][0]['stop'], int)
    
    