# PIkapiKA

## Usage

Run learner:

```
python learner.py --actor-num=1 --cuda
```

Run actor `actor-num` times:

```
python actor.py --render --cuda --benchmark --actor-id=0
```

* `actor-id`: start from 0 to actor-num - 1
* `benchmark`: print screenshot counter in 5 sec. x 10
