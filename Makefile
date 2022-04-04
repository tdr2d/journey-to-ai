export PYTHONPATH=.

help: ## Show this help
	@fgrep -h "##" Makefile | fgrep -v "fgrep" | sed -E 's/(.*):.*##(.*)/\1:\2/' | column -s: -t | sed -e 's/##//'
##
## Utils
sync: ## sync
	rsync -azP --exclude={.git/,build}./ user@192.168.95.150:/root/engine


##
## Modules
sudoku: ## 00 sudoku
	python3 modules/00_sudoku/sudoku-detector.py