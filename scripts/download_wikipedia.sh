DATA_DIR=./data/wikipedia
mkdir -p $DATA_DIR
cd $DATA_DIR

# download wikipedia
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2


# extract wikipedia
bzip2 -dkv enwiki-latest-pages-articles.xml.bz2
git clone https://github.com/attardi/wikiextractor
python3 wikiextractor/WikiExtractor.py -o wiki_text enwiki-latest-pages-articles.xml

