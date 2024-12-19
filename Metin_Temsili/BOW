from sklearn.feature_extraction.text import CountVectorizer

documets = ['Kedi at home',
            'Kedi bizim evde',
            'Bizim evde at'
            ]
vectorizer = CountVectorizer()

# metin -> sayısal vectör
X = vectorizer.fit_transform(documets)

# kelime kümesi 
print("kelime kümesi: ", vectorizer.get_feature_names_out())

# vector temsili
print("vector temsili: ", X.toarray())

"""
Her bir satır bir belgeyi(yani metni) temsil eder. Her sütun, bir kelimenin belgede kaç kez geçtiğini gösterir.

kelime kümesi:  ['at' 'bizim' 'evde' 'home' 'kedi']
vector temsili: 
 [
  [1 0 0 1 1]
  [0 1 1 0 1]
  [1 1 1 0 0]
]
"""