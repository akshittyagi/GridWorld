run:
	python DecisionProcess.py
run_debug:
	python DecisionProcess.py > debugFile
clean:
	rm *.pyc
clean_all:
	rm *.csv
	rm *.pkl
	rm *.pyc