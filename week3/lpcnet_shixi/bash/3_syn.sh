##########################################################################
# File Name: syn.sh
# Author: zcwang
# Created Time: 2020年01月03日 星期五 21时06分42秒
#########################################################################
#!/bin/bash
#example: bash bash/3_syn.sh dir_mel dir_out None/test.list(the same format as file.list)

#please use the absolute path
#NB: the mel file has to be f32 file in numpy array(*.mel),maybe you should use mel_format.py to covert your format before this step`
#Notice: the range of mel value should be in (-4,4)
dir_mel=$1
dir_out=$2
testlist=$3
stage=1

if [ $stage -le 1 ]; then
	python bash/mel_format.py ${dir_mel} temp \
	&& dir_mel=temp 
fi

#lpcnet_demo
if [ $stage -le 2 ]; then
	python scripts/lpcnet_demo.py $dir_mel $dir_out $testlist
fi
rm -rf $dir_mel
