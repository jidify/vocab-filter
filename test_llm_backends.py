import json
import unittest
from unittest.mock import patch

from pipeline import config, llm


class _Response:
    status = 200
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.body).encode()


class LLMBackendTests(unittest.TestCase):
    def test_catgpt_openai_compatible_request(self):
        with patch.multiple(config, LLM_BACKEND="catgpt", CATGPT_BASE_URL="http://gateway/v1",
                            CATGPT_API_TOKEN="secret", CATGPT_MODEL="catgpt-browser",
                            ), \
             patch("pipeline.llm._cache_path") as cache_path, \
             patch("pipeline.llm.urllib.request.urlopen") as open_url:
            cache_path.return_value.exists.return_value = False
            open_url.return_value = _Response({"choices": [{"message": {"content": '{"ok": true}'}}]})
            self.assertEqual(llm.call_json("question", system="consigne"), {"ok": True})
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, "http://gateway/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            payload = json.loads(request.data)
            self.assertEqual(payload["messages"], [
                {"role": "system", "content": "consigne"},
                {"role": "user", "content": "question"},
            ])
            self.assertFalse(payload["stream"])

    def test_ollama_protocol_is_preserved(self):
        with patch.multiple(config, LLM_BACKEND="ollama", OLLAMA_URL="http://ollama",
                            OLLAMA_MODEL="model"), \
             patch("pipeline.llm._cache_path") as cache_path, \
             patch("pipeline.llm.urllib.request.urlopen") as open_url:
            cache_path.return_value.exists.return_value = False
            open_url.return_value = _Response({"response": '{"ok": true}'})
            self.assertEqual(llm.call_json("question"), {"ok": True})
            self.assertEqual(open_url.call_args.args[0].full_url, "http://ollama/api/generate")


if __name__ == "__main__":
    unittest.main()
