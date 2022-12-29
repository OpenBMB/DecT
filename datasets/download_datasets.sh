#!/bin/sh

python download_glue_data.py --data_dir ./ --tasks MNLI,RTE

mv MNLI mnli
mv RTE rte

rm -rf agnews
wget --content-disposition https://cloud.tsinghua.edu.cn/f/0fb6af2a1e6647b79098/?dl=1
tar -zxvf agnews.tar.gz
rm -rf agnews.tar.gz

rm -rf dbpedia
wget --content-disposition https://cloud.tsinghua.edu.cn/f/362d3cdaa63b4692bafb/?dl=1
tar -zxvf dbpedia.tar.gz
rm -rf dbpedia.tar.gz

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

rm -rf SST-2
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bccfdb243eca404f8bf3/?dl=1
tar -zxvf SST-2.tar.gz
mv SST-2 sst2
rm -rf SST-2.tar.gz

rm -rf FewNERD
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bcacdddd54c44c5e86b1/?dl=1
tar -zxvf FewNERD.tar.gz
mv FewNERD fewnerd
rm -rf FewNERD.tar.gz

rm -rf snli
wget --content-disposition https://cloud.tsinghua.edu.cn/f/a1cf011883a845c7833f/?dl=1
unzip snli.zip
rm -rf snli.zip

rm -rf yelp
wget --content-disposition https://cloud.tsinghua.edu.cn/f/f3c8714d6a5c4b97b612/?dl=1
tar zxvf yelp_review_polarity_csv.tgz
mv yelp_review_polarity_csv yelp
rm -rf yelp_review_polarity_csv.tgz

rm -rf yahoo
wget --content-disposition https://cloud.tsinghua.edu.cn/f/640cb6cdc1864f89bd58/?dl=1
unzip yahoo.zip
rm -rf yahoo.zip

cd ..
