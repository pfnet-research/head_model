import contextlib
import copy
import io
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.util import dann, mlp, trainer
from app.util.data import Dataset


def make_dataset(classes, seed):
    random = torch.Generator().manual_seed(seed)
    features = torch.randn(24, 6, generator=random)
    domains = torch.arange(24).remainder(2).float()
    labels = torch.arange(24).remainder(classes)
    return Dataset(features, domains, labels)


def make_model(classes=3, dropout=True):
    model = dann.get_dann(
        D=6, K=classes, d_hidden=16, d_latent=8, n_layers=1,
        discriminator_loss_weight=0.7,
    )
    for layer in model.modules():
        if isinstance(layer, mlp.MLP):
            layer.dropout = dropout
    return model


def options(batch_size=6, lr=0.02, optimizer_type="sgd", n_epochs=2):
    return {
        "device": "cpu", "batch_size": batch_size, "lr": lr, "n_epochs": n_epochs,
        "optimizer_type": optimizer_type, "objective_domain": "target",
    }


class TestDANNTrainingCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.random.default_generator.manual_seed(77)

    def test_epoch_gradient_matches_only_the_last_shuffled_batch(self):
        dataset = make_dataset(3, seed=91)
        model = make_model(dropout=False)
        oracle_model = copy.deepcopy(model)
        args = options(lr=0.0)
        runner = trainer.Trainer(model, **args)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        batches = []
        real_loader = torch.utils.data.DataLoader

        class RecordingLoader(real_loader):
            def __iter__(self):
                for batch in super().__iter__():
                    batches.append(batch)
                    yield batch

        with patch.object(torch.utils.data, "DataLoader", RecordingLoader):
            runner._train_one_epoch(dataset, optimizer, args)

        self.assertEqual(len(batches), 4)
        expected = torch.autograd.grad(
            oracle_model(*batches[-1]), tuple(oracle_model.parameters()), allow_unused=True,
        )
        for param, grad in zip(model.parameters(), expected):
            if grad is None:
                self.assertIsNone(param.grad)
            else:
                torch.testing.assert_close(param.grad, grad, atol=1e-7, rtol=1e-6)

    def test_dropout_is_deterministic_only_in_evaluation(self):
        dataset = make_dataset(3, seed=19)
        model = make_model()
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(dataset.X, dataset.z)
            labels = model.predict(dataset.X, dataset.z)
            for _ in range(3):
                torch.testing.assert_close(
                    model.encode(dataset.X, dataset.z), embeddings, atol=0, rtol=0,
                )
                self.assertTrue(torch.equal(model.predict(dataset.X, dataset.z), labels))

    def test_training_still_applies_dropout(self):
        net = mlp.MLP([8, 8, 8])
        tensor = torch.ones(16, 8)
        net.train()
        self.assertFalse(torch.equal(net(tensor), net(tensor)))

    def test_every_optimizer_update_uses_training_mode(self):
        datasets = tuple(make_dataset(3, seed) for seed in (4, 5, 6))
        model = make_model()
        args = options()
        runner = trainer.Trainer(model, **args)
        events = []
        original_step = torch.optim.SGD.step
        original_evaluate = runner._evaluate

        def step(optimizer, *args, **kwargs):
            events.append(("step", model.training))
            return original_step(optimizer, *args, **kwargs)

        def evaluate(dataset, args):
            metrics = original_evaluate(dataset, args)
            phase = "validation" if dataset is datasets[1] else "test"
            events.append((phase, model.training))
            return metrics

        with patch.object(torch.optim.SGD, "step", step), patch.object(
            runner, "_evaluate", evaluate,
        ), contextlib.redirect_stdout(io.StringIO()):
            list(runner.train(datasets))

        self.assertEqual(
            events, ([("step", True)] * 4 + [("validation", False), ("test", False)]) * 2,
        )
        self.assertFalse(model.training)

    def test_predict_proba_matches_logits_in_both_domains(self):
        for classes, dropout in ((3, True), (5, False)):
            with self.subTest(classes=classes, dropout=dropout), torch.no_grad():
                dataset = make_dataset(classes, seed=16)
                model = make_model(classes, dropout)
                model.eval()
                source = dataset.z.bool()
                embeddings = torch.where(
                    source[:, None], model.encoder1(dataset.X), model.encoder2(dataset.X),
                )
                expected = model.predictor(embeddings).softmax(dim=1)
                model.train()
                actual = model.predict_proba(dataset.X, dataset.z)
                self.assertEqual(actual.shape, (len(dataset), classes))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual.sum(dim=1), torch.ones(len(dataset)))
                for mask in (source, ~source):
                    torch.testing.assert_close(actual[mask], expected[mask])
                torch.testing.assert_close(
                    model.predict_proba(dataset.X, dataset.z), actual, atol=0, rtol=0,
                )
                self.assertTrue(torch.equal(model.predict(dataset.X, dataset.z), actual.argmax(1)))

    def test_loss_terms_match_the_same_supervised_and_domain_objectives(self):
        model = make_model(dropout=False)
        X, z, labels = make_dataset(3, seed=21).tensors
        embeddings = model.encode(X, z)
        expected_supervised = F.cross_entropy(model.predictor(embeddings), labels)
        expected_domain = F.binary_cross_entropy_with_logits(
            model.discriminator(embeddings).squeeze(), z,
        )
        actual = model(X, z, labels)
        torch.testing.assert_close(model.loss_supervised, expected_supervised)
        torch.testing.assert_close(model.loss_unsupervised, expected_domain)
        torch.testing.assert_close(actual, expected_supervised + 0.7 * expected_domain)

    def test_optimizer_updates_and_metrics_remain_finite(self):
        for classes, batch_size, kind, optimizer in (
            (3, 4, "sgd", torch.optim.SGD), (5, 8, "adam", torch.optim.Adam),
        ):
            with self.subTest(classes=classes, batch_size=batch_size, optimizer=kind):
                datasets = tuple(make_dataset(classes, seed) for seed in (101, 102, 103))
                original = [tuple(value.clone() for value in data.tensors) for data in datasets]
                model = make_model(classes, dropout=kind == "adam")
                before = [param.detach().clone() for param in model.parameters()]
                runner = trainer.Trainer(model, **options(batch_size, optimizer_type=kind))
                self.assertIs(trainer._get_optimizer(kind), optimizer)
                with patch.object(trainer, "_get_optimizer", wraps=trainer._get_optimizer) as get:
                    with contextlib.redirect_stdout(io.StringIO()):
                        scores = list(runner.train(datasets))
                    get.assert_called_once_with(kind)
                self.assertEqual(len(scores), 2)
                self.assertTrue(all(math.isfinite(score) for score in scores))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                for result in runner.result:
                    train_losses = result["train"]["loss"]
                    self.assertEqual(set(train_losses), {"total", "supervised", "unsupervised"})
                    self.assertTrue(all(math.isfinite(loss) for loss in train_losses.values()))
                    self.assertAlmostEqual(
                        train_losses["total"],
                        train_losses["supervised"] + 0.7 * train_losses["unsupervised"],
                        places=5,
                    )
                    for phase in ("val", "test"):
                        self.assertTrue(math.isfinite(result[phase]["loss"]["total"]))
                        for accuracy in result[phase]["acc"].values():
                            self.assertTrue(0 <= accuracy <= 1)
                for data, originals in zip(datasets, original):
                    for actual, expected in zip(data.tensors, originals):
                        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
