from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['127.0.0.1'],
    port=9200
)

docs = [
    {
        # Zeppelin-Staaken R. VI
        'question': 'First flight Zeppelin-Staaken R.VI',
        'answer': 'The first Zeppelin-Staaken R.VI flight took place in 1916'
    }, {
        'question': 'Type Zeppelin-Staaken R.VI',
        'answer': 'The Zeppelin-Staaken R.VI aircraft type is bomber'
    }, {
        'question': 'Built Zeppelin-Staaken R.VI',
        'answer': 'The Zeppelin-Staaken R.VI was built 56 times.'
    }, {
        'question': 'Length Zeppelin-Staaken R.VI',
        'answer': 'The length Zeppelin-Staaken R.VI is 22.1 m'
    }, {
        'question': 'Span Zeppelin-Staaken R.VI',
        'answer': 'The span Zeppelin-Staaken R.VI is 42.2 m'
    },  {
        'question': 'The maximum take-off weight TMOW Zeppelin-Staaken R.VI',
        'answer': 'The maximum take-off weight of the Zeppelin-Staaken R.VI is 11.8 t.'
    }, {
        # Dornier Do X
        'question': 'First flight Dornier Do X',
        'answer': 'The first Dornier Do X flight took place in 12 Jul 1929'
    }, {
        'question': 'Type Dornier Do X',
        'answer': 'The Dornier Do X aircraft type is Flying boat'
    }, {
        'question': 'Built Dornier Do X',
        'answer': 'The Dornier Do X was built 3 times.'
    }, {
        'question': 'Length Dornier Do X',
        'answer': 'The length Dornier Do X is 40 m'
    }, {
        'question': 'Span Dornier Do X',
        'answer': 'The span Dornier Do X is 47.8 m'
    },  {
        'question': 'The maximum take-off weight TMOW Dornier Do X',
        'answer': 'The maximum take-off weight of the Dornier Do X is 52 t.'
    }, {
        # Kalinin K-7
        'question': 'First flight Kalinin K-7',
        'answer': 'The first Kalinin K-7 flight took place in 11 Aug 1933'
    }, {
        'question': 'Type Kalinin K-7',
        'answer': 'The Kalinin K-7 aircraft type is Transport'
    }, {
        'question': 'Built Kalinin K-7',
        'answer': 'The Kalinin K-7 was built 1 times.'
    }, {
        'question': 'Length Kalinin K-7',
        'answer': 'The length Kalinin K-7 is 28 m'
    }, {
        'question': 'Span Kalinin K-7',
        'answer': 'The span Kalinin K-7 is 53 m'
    },  {
        'question': 'The maximum take-off weight TMOW Kalinin K-7',
        'answer': 'The maximum take-off weight of the Kalinin K-7 is 46.5 t.'
    }, {
        # Tupolev ANT-20
        'question': 'First flight Tupolev ANT-20',
        'answer': 'The first Tupolev ANT-20 flight took place in 19 May 1934'
    }, {
        'question': 'Type Tupolev ANT-20',
        'answer': 'The Tupolev ANT-20 aircraft type is Transport'
    }, {
        'question': 'Built Tupolev ANT-20',
        'answer': 'The Tupolev ANT-20 was built 2 times.'
    }, {
        'question': 'Length Tupolev ANT-20',
        'answer': 'The length Tupolev ANT-20 is 32.9 m'
    }, {
        'question': 'Span Tupolev ANT-20',
        'answer': 'The span Tupolev ANT-20 is 63 m'
    },  {
        'question': 'The maximum take-off weight TMOW Tupolev ANT-20',
        'answer': 'The maximum take-off weight of the Tupolev ANT-20 is 53 t.'
    }, {
        # Douglas XB-19
        'question': 'First flight Douglas XB-19',
        'answer': 'The first Douglas XB-19 flight took place in 27 Jun 1941'
    }, {
        'question': 'Type Douglas XB-19',
        'answer': 'The Douglas XB-19 aircraft type is Bomber'
    }, {
        'question': 'Built Douglas XB-19',
        'answer': 'The Douglas XB-19 was built 1 times.'
    }, {
        'question': 'Length Douglas XB-19',
        'answer': 'The length Douglas XB-19 is 40.3 m'
    }, {
        'question': 'Span Douglas XB-19',
        'answer': 'The span Douglas XB-19 is 64.6 m'
    },  {
        'question': 'The maximum take-off weight TMOW Douglas XB-19',
        'answer': 'The maximum take-off weight of the Douglas XB-19 is 73.5 t.'
    }, {
        # Messerschmitt Me 323
        'question': 'First flight Messerschmitt Me 323',
        'answer': 'The first Messerschmitt Me 323 flight took place in 20 Jan 1942'
    }, {
        'question': 'Type Messerschmitt Me 323',
        'answer': 'The Messerschmitt Me 323 aircraft type is Transport'
    }, {
        'question': 'Built Messerschmitt Me 323',
        'answer': 'The Messerschmitt Me 323 was built 198 times.'
    }, {
        'question': 'Length Messerschmitt Me 323',
        'answer': 'The length Messerschmitt Me 323 is 28.2 m'
    }, {
        'question': 'Span Messerschmitt Me 323',
        'answer': 'The span Messerschmitt Me 323 is 55.2 m'
    },  {
        'question': 'The maximum take-off weight TMOW Messerschmitt Me 323',
        'answer': 'The maximum take-off weight of the Messerschmitt Me 323 is 43 t.'
    }, {
        # Martin JRM Mars
        'question': 'First flight Martin JRM Mars',
        'answer': 'The first Martin JRM Mars flight took place in 23 Jun 1942'
    }, {
        'question': 'Type Martin JRM Mars',
        'answer': 'The Martin JRM Mars aircraft type is Flying boat'
    }, {
        'question': 'Built Martin JRM Mars',
        'answer': 'The Martin JRM Mars was built 7 times.'
    }, {
        'question': 'Length Martin JRM Mars',
        'answer': 'The length Martin JRM Mars is 35.7 m'
    }, {
        'question': 'Span Martin JRM Mars',
        'answer': 'The span Martin JRM Mars is 61 m'
    },  {
        'question': 'The maximum take-off weight TMOW Martin JRM Mars',
        'answer': 'The maximum take-off weight of the Martin JRM Mars is 74.8 t.'
    }, {
        # Latécoère 631
        'question': 'First flight Latécoère 631',
        'answer': 'The first Latécoère 631 flight took place in 4 Nov 1942'
    }, {
        'question': 'Type Latécoère 631',
        'answer': 'The Latécoère 631 aircraft type is Flying boat'
    }, {
        'question': 'Built Latécoère 631',
        'answer': 'The Latécoère 631 was built 11 times.'
    }, {
        'question': 'Length Latécoère 631',
        'answer': 'The length Latécoère 631 is 43.5 m'
    }, {
        'question': 'Span Latécoère 631',
        'answer': 'The span Latécoère 631 is 57.4 m'
    },  {
        'question': 'The maximum take-off weight TMOW Latécoère 631',
        'answer': 'The maximum take-off weight of the Latécoère 631 is 71.4 t.'
    }, {
        # Junkers Ju 390
        'question': 'First flight Junkers Ju 390',
        'answer': 'The first Junkers Ju 390 flight took place in 20 Oct 1943'
    }, {
        'question': 'Type Junkers Ju 390',
        'answer': 'The Junkers Ju 390 aircraft type is Bomber'
    }, {
        'question': 'Built Junkers Ju 390',
        'answer': 'The Junkers Ju 390 was built 2 times.'
    }, {
        'question': 'Length Junkers Ju 390',
        'answer': 'The length Junkers Ju 390 is 34.2 m'
    }, {
        'question': 'Span Junkers Ju 390',
        'answer': 'The span Junkers Ju 390 is 50.3 m'
    },  {
        'question': 'The maximum take-off weight TMOW Junkers Ju 390',
        'answer': 'The maximum take-off weight of the Junkers Ju 390 is 75.5 t.'
    }, {
        # Blohm & Voss BV 238
        'question': 'First flight Blohm & Voss BV 238',
        'answer': 'The first Blohm & Voss BV 238 flight took place in Apr 1944'
    }, {
        'question': 'Type Blohm & Voss BV 238',
        'answer': 'The Blohm & Voss BV 238 aircraft type is Flying boat'
    }, {
        'question': 'Built Blohm & Voss BV 238',
        'answer': 'The Blohm & Voss BV 238 was built 1 times.'
    }, {
        'question': 'Length Blohm & Voss BV 238',
        'answer': 'The length Blohm & Voss BV 238 is 43.3 m'
    }, {
        'question': 'Span Blohm & Voss BV 238',
        'answer': 'The span Blohm & Voss BV 238 is 60.2 m'
    },  {
        'question': 'The maximum take-off weight TMOW Blohm & Voss BV 238',
        'answer': 'The maximum take-off weight of the Blohm & Voss BV 238 is 100 t.'
    }
]

for index, doc in enumerate(docs):
    res = es.index(index='python-pln-elasticsearch', id=index, body=doc)
    print('status: ' + res['result']+'. Index: ' + str(index))
