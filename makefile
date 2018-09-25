run:
	python DecisionProcess.py -e $(episodes) -p $(policy)
run_debug:
	python DecisionProcess.py > debugFile
clean:
	rm *.pyc
	rm debug*
clean_all:
	rm *.txt
	rm *.csv
	rm *.pkl
	rm *.pyc
	rm debug*
