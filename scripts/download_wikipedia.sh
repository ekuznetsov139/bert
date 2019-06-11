set -e
mkdir -p bert_data
cd bert_data

# download wikipedia
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2


# extract wikipedia
bzip2 -dkv enwiki-latest-pages-articles.xml.bz2 # > enwiki-latest-pages-articles.xml
git clone https://github.com/attardi/wikiextractor
python wikiextractor/WikiExtractor.py -o wikipedia enwiki-latest-pages-articles.xml

