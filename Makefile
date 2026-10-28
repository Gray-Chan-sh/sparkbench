profile ?= gb10

.PHONY: smoke a b c d e bench report audit pull push shots clean

smoke:
	python -m sparkbench smoke --profile $(profile)

a b c d e:
	python -m sparkbench $@ --profile $(profile)

bench:
	python -m sparkbench all --profile $(profile)

report:
	python tools/report.py --profile $(profile)

audit:
	python -m sparkbench audit

shots:
	python -m sparkbench shots

pull:
	python tools/pull.py --host $(host) --port $(port) --interval 600

push:
	python tools/push.py --git --webhook

run:
	bash run.sh

offers:
	python tools/vast.py offers --gpu GB10

watch:
	python tools/vast.py watch --gpu GB10 --interval 60

clean:
	rm -rf results
