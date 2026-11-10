.PHONY: train robust attack demo test clean run

train:
	PYTHONPATH=. python src/model.py

robust:
	PYTHONPATH=. python src/train_robust.py --epochs 5

attack:
	PYTHONPATH=. python src/run_experiments.py

demo:
	PYTHONPATH=. python src/demo.py

run:
	PYTHONPATH=. python src/model.py
	PYTHONPATH=. python src/train_robust.py --epochs 5
	PYTHONPATH=. python src/run_experiments.py

test:
	PYTHONPATH=. pytest test/ -v

clean:
	rm -rf models/ reports/figures/*.png data/MNIST
	@echo "Cleaned."
