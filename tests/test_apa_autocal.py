"""Tests for scripts/apa_autocal.py.

  python -m pytest tests            (or: python -m unittest discover tests)

The macro-parity test renders APA_FINISH_CELL from adaptive_pa_macro.cfg with
jinja2 and the end-to-end test runs against a mock Moonraker built on
websockets; each is skipped when its package is missing.
"""

import asyncio
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "apa_autocal", os.path.join(ROOT, "scripts", "apa_autocal.py"))
apa = importlib.util.module_from_spec(_spec)
sys.modules["apa_autocal"] = apa
_spec.loader.exec_module(apa)

try:
    import jinja2
except ImportError:
    jinja2 = None
try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    websockets = None

# Real FINISH inputs from a T2 suite at 265 C (Fluidd console history)
REAL_FINISH = [
    (0.015, 12740, 0.020, -3934, 45768),
    (0.015, 16165, 0.020, -2052, 39442),
    (0.015, 558, 0.020, -1139, 6486),
    (0.030, 15350, 0.035, -78431, 378535),
    (0.020, 2806, 0.025, -7475, 43836),
]


class ParseTests(unittest.TestCase):
    def test_firmware_format(self):
        self.assertEqual(apa.parse_area("// k0.015: area=-5421.000"), (0.015, -5421.0))

    def test_colon_format(self):
        self.assertEqual(apa.parse_area("k0.010: area: +1278"), (0.010, 1278.0))

    def test_other_lines(self):
        self.assertIsNone(apa.parse_area("B:28.1 /0.0 T0:29.9 /220.0"))
        self.assertIsNone(apa.parse_area("// k min: 0.015, max: 0.025, step: 0.005"))

    def test_expected_ks_exclusive_max(self):
        self.assertEqual(apa.expected_ks(0.005, 0.040, 0.005),
                         [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
        self.assertEqual(apa.expected_ks(0.015, 0.025, 0.005), [0.015, 0.02])
        self.assertEqual(apa.expected_ks(0.016, 0.020, 0.001),
                         [0.016, 0.017, 0.018, 0.019])

    def test_expected_ks_rejects_sub_thousandth(self):
        with self.assertRaises(ValueError):
            apa.expected_ks(0.01, 0.02, 0.0025)

    def test_flow_log(self):
        log = (
            "2026-10-20 11:56:53 [INFO] Received request: loop=14, slowv=5.078, fastv=6.770, "
            "pt_len=28118\n"
            "2026-10-20 11:56:53 [INFO] Calculation result: -5421\n"
            "2026-10-20 11:57:00 [INFO] Received request: loop=14, slowv=2.539, fastv=3.385\n"
            "2026-10-20 11:57:01 [INFO] Calculation result: 999\n"
            "2026-10-20 11:57:32 [INFO] Received request: loop=14, slowv=5.078, fastv=6.770\n"
            "2026-10-20 11:57:32 [INFO] Calculation result: -18864\n")
        fast = 200 * 0.033851
        self.assertEqual(apa.flow_log_areas(log, fast * 0.75, fast), [-5421.0, -18864.0])

    def test_console_log(self):
        log = "\n".join([
            "// ========== APA TEST POINT START ==========",
            "// Point: center | Tool T2 (extruder2) | MODE=MEASURE TEMP=265",
            "// XY: speed=200.0 mm/s  accel=6000.0  ->  Q=16.2832 mm3/s",
            "// k0.010: area=30500.000",
            "// k0.015: area=12740.000",
            "// k0.020: area=-3934.000",
            "// ========== APA TEST POINT MEASURE DONE ==========",
        ])
        (p,) = apa.parse_console_log(log)
        self.assertEqual((p["name"], p["tool"], p["speed"], p["accel"], p["flow"]),
                         ("center", 2, 200.0, 6000.0, 16.2832))
        self.assertEqual(p["samples"], {0.01: 30500.0, 0.015: 12740.0, 0.02: -3934.0})


class MathTests(unittest.TestCase):
    def test_bracket_and_score_user_guide_example(self):
        samples = {0.010: 12450, 0.015: 5128, 0.020: -5281, 0.025: -11800}
        r = apa.evaluate(samples)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["bracket"], [0.015, 5128, 0.020, -5281])
        self.assertEqual(r["k_out"], 0.017)
        self.assertAlmostEqual(r["k_star"], 0.017463, places=5)
        self.assertEqual((r["confidence"], r["grade"], r["clean_zero"]), (83, "excellent", False))

    def test_no_flip_statuses(self):
        self.assertEqual(apa.find_bracket({0.01: 5, 0.02: 3})[1], "kstar_above")
        self.assertEqual(apa.find_bracket({0.015: -5421, 0.02: -18864})[1], "kstar_below")
        self.assertEqual(apa.find_bracket({0.01: -5, 0.02: 3})[1], "no_flip")
        self.assertEqual(apa.find_bracket({})[1], "no_data")

    def test_exact_zero(self):
        r = apa.evaluate({0.01: 500, 0.015: 0, 0.02: -700})
        self.assertEqual(r["k_star"], 0.015)

    def test_noisy_note(self):
        r = apa.evaluate({0.01: 500, 0.015: -100, 0.02: 50, 0.025: -700})
        self.assertEqual(r["bracket"][:3:2], [0.01, 0.015])
        self.assertTrue(any("noisy" in n for n in r["notes"]))

    def test_retry_ranges(self):
        above = {k: 1000.0 for k in apa.expected_ks(0.005, 0.040, 0.005)}
        lo, hi, st = apa.retry_range(above, "kstar_above", 0.005, 0.040, 0.005)
        self.assertEqual(apa.expected_ks(lo, hi, st)[:2], [0.04, 0.045])
        below = {0.015: -5421.0, 0.02: -18864.0}
        lo, hi, st = apa.retry_range(below, "kstar_below", 0.015, 0.025, 0.005)
        self.assertEqual(apa.expected_ks(lo, hi, st), [0.005, 0.01])
        self.assertIsNone(apa.retry_range({0.0: -1.0}, "kstar_below", 0.0, 0.01, 0.005))

    def test_refine_range(self):
        fine = apa.refine_range([0.015, 558, 0.020, -1139], 0.005)
        self.assertEqual(apa.expected_ks(*fine), [0.016, 0.017, 0.018, 0.019])
        self.assertIsNone(apa.refine_range([0.015, 1, 0.016, -1], 0.001))

    def test_fallback_median(self):
        pts = [{"status": "ok", "k_out": k, "confidence": 80}
               for k in (0.012, 0.013, 0.016, 0.017, 0.023)]
        self.assertEqual(apa.fallback_pa(pts), 0.016)
        pts.append({"status": "ok", "k_out": 0.5, "confidence": 20})
        self.assertEqual(apa.fallback_pa(pts), 0.016)

    def test_orca_row(self):
        self.assertEqual(apa.orca_row(0.017, 8.1416, 2000.0), "0.017, 8.14, 2000")


def _macro_template(name):
    with open(os.path.join(ROOT, "adaptive_pa_macro.cfg"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"\[gcode_macro %s\].*?\ngcode:\n(.*?)(?=\n\[|\Z)" % re.escape(name),
                  text, re.S)
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in m.group(1).splitlines())
    env = jinja2.Environment("{%", "%}", "{", "}")
    return env.from_string(body)


@unittest.skipIf(jinja2 is None, "jinja2 not installed")
class MacroParityTests(unittest.TestCase):
    """The Python score() must match APA_FINISH_CELL in adaptive_pa_macro.cfg."""

    def run_macro(self, k0, a0, k1, a1, max_abs):
        out = []
        tpl = _macro_template("APA_FINISH_CELL")
        tpl.render(
            printer={"gcode_macro _APA_COIL_CFG": {"pending_valid": 0}},
            params={"K0": str(k0), "A0": str(a0), "K1": str(k1), "A1": str(a1),
                    "MAX_ABS": str(max_abs), "FLOW": "8.1416", "ACCEL": "2000", "NAME": "t"},
            action_respond_info=lambda s: out.append(s) or "",
            action_raise_error=lambda s: (_ for _ in ()).throw(ValueError(s)))
        text = "\n".join(out)
        conf = int(re.search(r"Confidence: (\d+)/100", text).group(1))
        k_out = float(re.search(r"rounded ([\d.]+)", text).group(1))
        return conf, k_out

    def check(self, k0, a0, k1, a1, max_abs):
        mine = apa.score(k0, a0, k1, a1, max_abs)
        self.assertEqual(self.run_macro(k0, a0, k1, a1, max_abs),
                         (mine["confidence"], mine["k_out"]),
                         (k0, a0, k1, a1, max_abs))

    def test_real_and_documented_inputs(self):
        self.check(0.015, 5128, 0.020, -5281, 12450)
        for case in REAL_FINISH:
            self.check(*case)

    def test_random_inputs(self):
        rnd = random.Random(1)
        for _ in range(400):
            k0 = rnd.randrange(0, 60) / 1000.0
            k1 = k0 + rnd.choice((0.001, 0.002, 0.005))
            a0 = rnd.choice((0, rnd.uniform(1, 4e5), rnd.uniform(1, 400)))
            a1 = -rnd.choice((rnd.uniform(1, 4e5), rnd.uniform(1, 900)))
            if a0 == 0 and a1 == 0:
                continue
            max_abs = rnd.choice((0, max(abs(a0), abs(a1)) * rnd.uniform(1, 30)))
            self.check(round(k0, 3), round(a0, 3), round(k1, 3), round(a1, 3),
                       round(max_abs, 3))


class mock_cfg:
    """Temporarily replace MacroBehaviourTests.CFG for one test."""

    def __init__(self, test, cfg):
        self.test, self.cfg = test, cfg

    def __enter__(self):
        self.saved, self.test.CFG = self.test.CFG, self.cfg

    def __exit__(self, *exc):
        self.test.CFG = self.saved


@unittest.skipIf(jinja2 is None, "jinja2 not installed")
class MacroBehaviourTests(unittest.TestCase):
    CFG = {"line_area": 0.081416, "fil_area": 2.405282, "ratio": 0.033849,
           "low_speed": 100.0, "mid_speed": 200.0, "high_speed": 250.0,
           "low_accel": 2000.0, "mid_accel": 6000.0, "high_accel": 10000.0,
           "temp": 220, "k_min": 0.005, "k_max": 0.04, "k_step": 0.005}

    def render(self, name, **params):
        info = []
        gcode = _macro_template(name).render(
            printer={"gcode_macro _APA_COIL_CFG": self.CFG},
            params={k: str(v) for k, v in params.items()},
            action_respond_info=lambda s: info.append(s) or "",
            action_raise_error=lambda s: (_ for _ in ()).throw(ValueError(s)))
        return gcode, info

    def set_vars(self, **params):
        gcode, info = self.render("APA_COIL_SET_ENVELOPE", **params)
        found = re.findall(r"VARIABLE=(\w+) VALUE=(\S+)", gcode)
        return {k: float(v) for k, v in found}, info

    def test_no_hash_in_macro_bodies(self):
        with open(os.path.join(ROOT, "adaptive_pa_macro.cfg"), encoding="utf-8") as f:
            bodies = re.findall(r"\ngcode:\n(.*?)(?=\n\[|\Z)", f.read(), re.S)
        self.assertTrue(bodies)
        for body in bodies:
            # column-0 "####" separators between sections are plain comments
            for line in body.splitlines():
                if line.startswith(" "):
                    self.assertNotIn("#", line)

    def test_macro_names_are_callable(self):
        # Klipper ends a command name at the first digit: APA_COIL_PRESET_U1_DEFAULT
        # would run as APA_COIL_PRESET_U1 with "_DEFAULT" as an argument.
        with open(os.path.join(ROOT, "adaptive_pa_macro.cfg"), encoding="utf-8") as f:
            names = re.findall(r"^\[gcode_macro (\S+)\]", f.read(), re.M)
        self.assertTrue(names)
        for name in names:
            self.assertRegex(name, r"^[A-Z_]+\d*$")

    def test_geometry_matches_documented_default(self):
        v, _ = self.set_vars(LAYER_HEIGHT=0.2, LINE_WIDTH=0.45, FIL_DIA=1.75)
        self.assertAlmostEqual(v["line_area"], 0.081416, places=6)
        self.assertAlmostEqual(v["fil_area"], 2.405282, places=6)
        self.assertAlmostEqual(v["ratio"], 0.033849, places=6)

    def test_geometry_other_nozzle(self):
        v, _ = self.set_vars(LAYER_HEIGHT=0.3, LINE_WIDTH=0.62)
        self.assertAlmostEqual(v["line_area"], 0.3 * 0.32 + 3.14159265 * 0.15 ** 2, places=6)

    def test_untouched_geometry_kept(self):
        v, _ = self.set_vars(HIGH_SPEED=230)
        self.assertEqual((v["line_area"], v["ratio"], v["high_speed"]), (0.081416, 0.033849, 230.0))

    def test_unrelated_params_keep_mid_point(self):
        cfg = dict(self.CFG, mid_speed=180.0, mid_accel=5000.0)
        with mock_cfg(self, cfg):
            v, _ = self.set_vars(LAYER_HEIGHT=0.2, LINE_WIDTH=0.45)
            self.assertEqual((v["mid_speed"], v["mid_accel"]), (180.0, 5000.0))
            v, _ = self.set_vars(HIGH_SPEED=300)
            self.assertEqual((v["mid_speed"], v["mid_accel"]), (200.0, 5000.0))
            v, _ = self.set_vars(HIGH_ACCEL=12000)
            self.assertEqual((v["mid_speed"], v["mid_accel"]), (180.0, 7000.0))

    def test_mvs_caps_high_speed(self):
        v, info = self.set_vars(MVS=16.3)
        self.assertEqual(v["high_speed"], 200.2)
        self.assertEqual(v["mid_speed"], 150.1)
        self.assertTrue(any("capped" in s for s in info))

    def test_mvs_below_low_speed_errors(self):
        with self.assertRaises(ValueError):
            self.set_vars(MVS=5)

    def test_point_macro_keeps_active_tool(self):
        gcode, _ = self.render("APA_COIL_HIGH_FLOW")
        self.assertNotIn("EXTRUDER=", gcode)
        gcode, _ = self.render("APA_COIL_HIGH_FLOW", EXTRUDER=2)
        self.assertIn("EXTRUDER=2", gcode)


# ---------------------------------------------------------------------------
# End-to-end against a mock Moonraker
# ---------------------------------------------------------------------------

ENVELOPE = {
    "line_area": 0.081416, "fil_area": 2.405282, "ratio": 0.033849, "temp": 220,
    "k_min": 0.005, "k_max": 0.04, "k_step": 0.005, "loop": 14, "slow_frac": 0.75,
    "low_speed": 100.0, "mid_speed": 200.0, "high_speed": 250.0,
    "low_accel": 2000.0, "mid_accel": 6000.0, "high_accel": 10000.0,
    "last_k_star": 0.0, "pending_valid": 0,
}
# name -> (true K*, area per 0.005 K); stress fails like a real macro error
MODEL = {
    "low_anchor": (0.0172, 10000.0),
    "high_flow": (0.0471, 40000.0),   # above the default range -> moved range
    "high_force": (0.0167, 150.0),    # peak < 1000 caps confidence at 44 -> fine scan
    "center": (0.0118, 8000.0),
}


class MockMoonraker:
    def __init__(self, drop_console_for=None):
        self.commands = []
        self.drop_console_for = drop_console_for
        self.flow_log = []
        self.port = None
        self._ready = threading.Event()
        self._stop = None

    def area(self, name, k):
        k_star, gain = MODEL[name]
        return round(gain * (k_star - k) / 0.005, 3)

    async def handler(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            method, params, rid = msg["method"], msg.get("params", {}), msg["id"]
            result, error = "ok", None
            if method == "server.info":
                result = {"klippy_state": "ready"}
            elif method == "printer.objects.query":
                result = {"status": {"gcode_macro _APA_COIL_CFG": ENVELOPE,
                                     "print_stats": {"state": "standby"}}}
            elif method == "printer.gcode.script":
                script = params["script"]
                self.commands.append(script)
                lines, error = self.run_gcode(script)
                for line in lines:
                    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "notify_gcode_response",
                                              "params": [line]}))
            reply = {"jsonrpc": "2.0", "id": rid}
            reply.update({"error": {"code": 400, "message": error}} if error else {"result": result})
            await ws.send(json.dumps(reply))

    def run_gcode(self, script):
        p = dict(re.findall(r"(\w+)=(\S+)", script))
        if script.startswith("APA_COIL_CELL"):
            if p["NAME"] == "stress":
                return [], "!! Expected active extruder extruder, got extruder1"
            fast = float(p["SPEED"]) * ENVELOPE["ratio"]
            lines = ["B:28.1 /0.0 C:27.1 /0.0 T0:29.9 /220.0"]
            for k in apa.expected_ks(float(p["MIN"]), float(p["MAX"]), float(p["STEP"])):
                a = self.area(p["NAME"], k)
                self.flow_log.append("[INFO] Received request: loop=14, slowv=%.3f, fastv=%.3f"
                                     % (fast * 0.75, fast))
                self.flow_log.append("[INFO] Calculation result: %d" % a)
                if p["NAME"] != self.drop_console_for:
                    lines.append("// k%.3f: area=%.3f" % (k, a))
            return lines, None
        if script.startswith("APA_FINISH_CELL"):
            s = apa.score(float(p["K0"]), float(p["A0"]), float(p["K1"]), float(p["A1"]),
                          float(p["MAX_ABS"]))
            return ["// K* (zero cross): %s  ->  rounded %s" % (s["k_star"], s["k_out"]),
                    "// Confidence: %d/100 (%s)" % (s["confidence"], s["grade"])], None
        return [], None

    def start(self):
        threading.Thread(target=asyncio.run, args=(self._main(),), daemon=True).start()
        self._ready.wait(10)

    async def _main(self):
        self._stop = asyncio.Event()
        async with serve(self.handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()


@unittest.skipIf(websockets is None, "websockets not installed")
class EndToEndTests(unittest.TestCase):
    def run_tool(self, mock, *extra):
        out = tempfile.mkdtemp()
        apa.main(["--host", "127.0.0.1:%d" % mock.port, "--out", out, "--tools", "0", *extra])
        (session,) = os.listdir(out)
        d = os.path.join(out, session)
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            return d, json.load(f)

    def test_full_suite(self):
        mock = MockMoonraker()
        mock.start()
        d, res = self.run_tool(mock)
        pts = {p["name"]: p for p in res["tools"]["0"]["points"]}

        self.assertEqual(pts["low_anchor"]["k_out"], 0.017)
        self.assertEqual(len(pts["low_anchor"]["sweeps"]), 1)
        self.assertEqual(pts["center"]["k_out"], 0.012)

        hf = pts["high_flow"]
        self.assertEqual(hf["status"], "ok")
        self.assertEqual(len(hf["sweeps"]), 2)
        self.assertEqual(hf["sweeps"][1]["min"], 0.04)
        self.assertEqual(hf["k_out"], 0.047)

        hfo = pts["high_force"]
        self.assertEqual(len(hfo["sweeps"]), 2)
        self.assertEqual(hfo["sweeps"][1]["step"], 0.001)
        self.assertEqual(hfo["k_out"], 0.017)

        self.assertEqual(pts["stress"]["status"], "error")
        self.assertTrue(all("macro reports" not in n for p in pts.values()
                            for n in p.get("notes", [])))

        self.assertEqual(res["tools"]["0"]["fallback_pa"], 0.017)
        with open(os.path.join(d, "orca_T0.txt"), encoding="utf-8") as f:
            orca = f.read()
        self.assertIn("0.017, 8.14, 2000", orca)
        self.assertIn("0.047, 20.35, 2000", orca)
        self.assertNotIn("8.14, 10000", orca)  # weak high_force row is left out
        self.assertTrue(mock.commands[-1].startswith("SET_HEATER_TEMPERATURE HEATER=extruder "))
        with open(os.path.join(d, "console.log"), encoding="utf-8") as f:
            self.assertNotIn("B:28.1", f.read())

    def test_dry_run_sends_no_motion(self):
        mock = MockMoonraker()
        mock.start()
        out = tempfile.mkdtemp()
        apa.main(["--host", "127.0.0.1:%d" % mock.port, "--out", out, "--dry-run"])
        self.assertEqual(os.listdir(out), [])
        self.assertTrue(all(c.startswith("SET_HEATER") for c in mock.commands))

    def test_resume_skips_finished_points(self):
        mock = MockMoonraker()
        mock.start()
        d, _ = self.run_tool(mock, "--points", "center")
        mock.commands.clear()
        apa.main(["--host", "127.0.0.1:%d" % mock.port, "--resume", d, "--points",
                  "center,low_anchor"])
        cells = [c for c in mock.commands if c.startswith("APA_COIL_CELL")]
        self.assertEqual(len(cells), 1)
        self.assertIn("NAME=low_anchor", cells[0])

    def test_missing_console_lines_come_from_flow_log(self):
        mock = MockMoonraker(drop_console_for="center")
        mock.start()
        orig = apa.Moonraker.get_text
        apa.Moonraker.get_text = lambda self, path: "\n".join(mock.flow_log)
        try:
            _, res = self.run_tool(mock, "--points", "center")
        finally:
            apa.Moonraker.get_text = orig
        (p,) = res["tools"]["0"]["points"]
        self.assertEqual((p["status"], p["k_out"]), ("ok", 0.012))


if __name__ == "__main__":
    unittest.main()
