from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pdm_net import PDMConfig, PDMNet
from pdm_net.data import VideoClipDataset
from pdm_net.engine import AlternatingTrainer, AnomalyScorer, Evaluator
from pdm_net.engine.evaluator import binary_auc
from pdm_net.losses import PDMLoss, compactness, dispersion
from pdm_net.models.convlstm import ConvLSTMCell
from pdm_net.models.prototype import DynamicPrototypeMemory
from pdm_net.models.residual import ResidualSequenceBuilder


def small_config():
    return PDMConfig(long_frames=5, short_frames=3, image_size=32,
                     base_channels=4, feature_channels=8, num_prototypes=5)


def make_batch():
    clip = torch.rand(2, 5, 3, 32, 32)
    return {"long_clip": clip, "short_clip": clip[:, :3], "target": torch.rand(2, 3, 32, 32)}


def snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def changed(before, module):
    return any(not torch.equal(before[name], p) for name, p in module.named_parameters())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)

    def test_residual_sign_and_length(self):
        x = torch.tensor([0.1, 0.8, 0.2]).reshape(1, 3, 1, 1, 1)
        signed = ResidualSequenceBuilder()(x).flatten()
        torch.testing.assert_close(signed, torch.tensor([0.7, -0.6]))
        torch.testing.assert_close(ResidualSequenceBuilder("absolute")(x).flatten(), signed.abs())

    def test_convlstm_gate_equations(self):
        cell = ConvLSTMCell(1, 1)
        with torch.no_grad():
            cell.gates.weight.zero_()
            cell.gates.bias.zero_()
        h, c = cell(torch.zeros(1, 1, 2, 2),
                    (torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0)))
        torch.testing.assert_close(c, torch.ones_like(c))
        torch.testing.assert_close(h, 0.5 * torch.tanh(c))

    def test_read_cosine_weights_and_query_gradients(self):
        memory = DynamicPrototypeMemory(3, 2)
        with torch.no_grad():
            memory.prototypes.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        features = torch.tensor([[[[2.]], [[1.]]]], requires_grad=True)
        result = memory(features)
        expected = torch.tensor([2., 1., -2.]) / (5 ** 0.5)
        torch.testing.assert_close(result.weights.flatten(), expected.softmax(0))
        self.assertEqual(result.indices.item(), 0)
        torch.testing.assert_close(result.weights.sum(-1), torch.ones(1, 1))
        result.recalled.square().sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertIsNone(memory.prototypes.grad)

    def test_write_matches_reference_and_keeps_empty_prototype(self):
        memory = DynamicPrototypeMemory(3, 2)
        bank = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        with torch.no_grad():
            memory.prototypes.copy_(bank)
        queries = torch.tensor([[2., 0.], [1., 1.], [0., 2.]])
        similarities = F.normalize(queries, dim=-1) @ bank.T
        # Reference computes Eq. 4 explicitly, without softmax or read weights.
        raw = similarities.exp()
        omega = raw / raw.sum(0, keepdim=True)
        expected = bank.clone()
        assigned = similarities.argmax(-1)
        for index in range(3):
            selected = assigned == index
            if selected.any():
                weights = omega[selected, index] / omega[selected, index].max()
                expected[index] = F.normalize(bank[index] + (weights[:, None] * queries[selected]).sum(0), dim=0)
        memory.write(queries.unsqueeze(0))
        torch.testing.assert_close(memory.prototypes, expected)
        self.assertTrue(torch.equal(memory.prototypes[2], bank[2]))
        torch.testing.assert_close(memory.prototypes.norm(dim=-1), torch.ones(3))

    def test_compactness_is_unsquared_and_dispersion_has_gradient(self):
        self.assertEqual(compactness(torch.tensor([[[3., 4.]]]), torch.zeros(1, 1, 2)).item(), 5)
        bank = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
        indices = torch.tensor([[0, 0, 1, 2]])
        value = dispersion(bank, indices)
        centered = bank - bank.mean(0)
        matrix = sum(p * torch.outer(v, v) for p, v in zip([0.5, 0.25, 0.25], centered))
        torch.testing.assert_close(value, -matrix.trace())
        value.backward()
        self.assertGreater(bank.grad.abs().sum().item(), 0)
        with torch.no_grad():
            updated = bank - 0.01 * bank.grad
        self.assertLess(dispersion(updated, indices).item(), value.item())

    def test_loss_scale_and_bank_gradient_ownership(self):
        model = PDMNet(small_config())
        batch = make_batch()
        output = model.forward_long(batch["short_clip"], batch["long_clip"])
        mean = PDMLoss(prediction_reduction="mean")(output, batch["target"], model.memory.prototypes, "long")
        summed = PDMLoss(prediction_reduction="sum_per_sample")(output, batch["target"], model.memory.prototypes, "long")
        torch.testing.assert_close(summed["prediction"], mean["prediction"] * 3 * 32 * 32)
        bank_grad = torch.autograd.grad(mean["total"], model.memory.prototypes, retain_graph=True)[0]
        spread_grad = torch.autograd.grad(0.1 * mean["dispersion"], model.memory.prototypes)[0]
        torch.testing.assert_close(bank_grad, spread_grad)

    def test_alternation_updates_only_intended_modules(self):
        model = PDMNet(small_config())
        trainer = AlternatingTrainer(model)
        batch = make_batch()
        short_before, long_before = snapshot(model.short_encoder), snapshot(model.long_encoder)
        predictor_before = snapshot(model.decoder)
        bank_before = model.memory.prototypes.detach().clone()
        trainer.step_long(batch)
        self.assertFalse(changed(short_before, model.short_encoder))
        self.assertTrue(changed(long_before, model.long_encoder))
        self.assertTrue(changed(predictor_before, model.decoder))
        self.assertFalse(torch.equal(bank_before, model.memory.prototypes))
        long_after = snapshot(model.long_encoder)
        bank_after = model.memory.prototypes.detach().clone()
        decoder_after = snapshot(model.decoder)
        trainer.step_short(batch)
        self.assertTrue(changed(short_before, model.short_encoder))
        self.assertFalse(changed(long_after, model.long_encoder))
        self.assertTrue(changed(decoder_after, model.decoder))
        self.assertTrue(torch.equal(bank_after, model.memory.prototypes))
        self.assertIsNone(model.memory.prototypes.grad)
        self.assertTrue(all(p.grad is None for p in model.long_encoder.parameters()))

    def test_inference_uses_only_short_branch_and_does_not_write(self):
        model = PDMNet(small_config()).eval()
        def fail_if_called(*args):
            raise AssertionError("Long encoder must not run during inference")
        hook = model.long_encoder.register_forward_pre_hook(fail_if_called)
        bank = model.memory.prototypes.detach().clone()
        with torch.no_grad():
            prediction = model(make_batch()["short_clip"]).prediction
        hook.remove()
        self.assertEqual(prediction.shape, (2, 3, 32, 32))
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
        self.assertTrue(torch.equal(bank, model.memory.prototypes))

    def test_checkpoint_resumes_identical_next_step(self):
        config = small_config()
        model = PDMNet(config)
        trainer = AlternatingTrainer(model)
        batch = make_batch()
        trainer.train_batch(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            trainer.save_checkpoint(path, completed_epochs=1)
            reloaded = PDMNet(config)
            restored = AlternatingTrainer(reloaded)
            self.assertEqual(restored.load_checkpoint(path), 1)
            self.assertEqual(restored.global_step, 1)
            trainer.train_batch(batch)
            restored.train_batch(batch)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
            mismatched = AlternatingTrainer(PDMNet(replace(config, score_weight=0.5)))
            with self.assertRaises(ValueError):
                mismatched.load_checkpoint(path)

    def test_video_boundaries_natural_sort_and_labels(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frames"
            labels = Path(directory) / "labels"
            labels.mkdir()
            for video, offset in (("video1", 0), ("video2", 100)):
                folder = root / video
                folder.mkdir(parents=True)
                for i in range(12):
                    Image.fromarray(np.full((8, 8, 3), i + offset, dtype=np.uint8)).save(folder / f"{i}.png")
                np.save(labels / f"{video}.npy", np.arange(12) % 2)
            training = VideoClipDataset(root, config)
            self.assertEqual(len(training), 14)
            self.assertEqual(training[6]["target_index"], 11)
            self.assertEqual(training[6]["target_file"], "11.png")
            self.assertEqual(training[7]["video_id"], "video2")
            self.assertEqual(training[7]["target_index"], 5)
            torch.testing.assert_close(training[7]["long_clip"][:, 0, 0, 0], torch.arange(100, 105) / 255)
            testing = VideoClipDataset(root, config, training=False, labels_dir=labels)
            self.assertNotIn("long_clip", testing[0])
            self.assertEqual(testing[0]["label"], 1)
            # Test loader must read only observed indices 0..2 plus target 5.
            calls = []
            read = testing._read
            def recording_read(path):
                calls.append(path.stem)
                return read(path)
            testing._read = recording_read
            testing[0]
            self.assertEqual(calls, ["0", "1", "2", "5"])
            with self.assertRaises(ValueError):
                VideoClipDataset(root, config, labels_dir=labels)
            model = PDMNet(config)
            bank = model.memory.prototypes.detach().clone()
            rows, metrics = Evaluator(model).run(DataLoader(testing, batch_size=3))
            self.assertEqual(metrics["num_scored_frames"], 14)
            self.assertEqual(metrics["num_videos"], 2)
            self.assertEqual(rows[0]["target_index"], 5)
            self.assertTrue(all(r["label"] == r["target_index"] % 2 for r in rows))
            self.assertTrue(all(0 <= r["anomaly_score"] <= 1 for r in rows))
            self.assertTrue(torch.equal(bank, model.memory.prototypes))
            self.assertTrue(model.training)

    def test_scoring_direction_constants_auc_ties(self):
        scorer = AnomalyScorer()
        np.testing.assert_allclose(scorer.combine_video([40, 30, 20], [0, 1, 2]), [0, 0.5, 1])
        np.testing.assert_allclose(scorer.combine_video([120, 120], [0, 0]), [0.6, 0.6])
        self.assertEqual(binary_auc([0, 1], [0, 1]), 1.0)
        self.assertEqual(binary_auc([0, 1], [1, 0]), 0.0)
        self.assertEqual(binary_auc([0, 1], [1, 1]), 0.5)
        self.assertIsNone(binary_auc([0, 0], [0, 1]))
        with self.assertRaises(ValueError):
            scorer.combine_video([float("nan")], [0])
        model = PDMNet(small_config()).eval()
        with torch.no_grad():
            output = model(make_batch()["short_clip"])
        psnr, _ = scorer.raw_scores(output, output.prediction)
        np.testing.assert_allclose(psnr, [120, 120])

    def test_invalid_shapes_and_config_fail_early(self):
        with self.assertRaises(ValueError):
            PDMConfig(short_frames=13)
        with self.assertRaises(ValueError):
            PDMConfig(image_size=30)
        with self.assertRaises(ValueError):
            PDMConfig(learning_rate=float("nan"))
        model = PDMNet(small_config())
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
