"""Focused behavior checks, not an answer-quality evaluation dataset."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import numpy as np
import rag


class RAGTest(unittest.TestCase):
    def test_followup_query_keeps_context_and_translates(self):
        with patch.object(rag, "complete", return_value="rice irrigation") as complete:
            query = rag.prepare_query("దాని గురించి?", [{"role": "user", "content": "rice irrigation"}])
        self.assertEqual(query, "rice irrigation")
        self.assertEqual(complete.call_args.args[1]["history"][0]["content"], "rice irrigation")

    def test_citations_and_requested_language(self):
        response = {"supported": True, "sections": [{"text": "నీటిని ఆదా చేయండి.", "source_ids": [1]}]}
        with patch.object(rag, "complete", return_value=json.dumps(response)) as complete:
            answer = rag.generate_answer("నీరు?", [{"source": "water.pdf", "page": 3, "text": "Save water."}], language="Telugu")
        self.assertEqual(answer, "నీటిని ఆదా చేయండి. [1]")
        self.assertEqual(complete.call_args.args[1]["answer_language"], "Telugu")

    def test_invalid_and_missing_citations_are_rejected(self):
        for ids in ([2], [], [0]):
            with self.subTest(ids=ids), patch.object(rag, "complete", return_value=json.dumps({"supported": True, "sections": [{"text": "Claim", "source_ids": ids}]})):
                with self.assertRaises(ValueError):
                    rag.generate_answer("q", [{"source": "s", "page": 1, "text": "t"}])

    def test_too_little_evidence_has_no_citation(self):
        with patch.object(rag, "complete", return_value=json.dumps({"supported": False, "sections": [{"text": "Insufficient information.", "source_ids": []}]})):
            self.assertEqual(rag.generate_answer("q", []), "Insufficient information.")

    def test_small_index_does_not_duplicate_negative_positions(self):
        index = faiss.IndexFlatL2(2)
        index.add(np.array([[0, 0]], dtype="float32"))
        base = {"index": index, "chunks": [{"source": "base", "page": 1, "text": "base text"}]}
        model = Mock()
        model.encode.return_value = np.array([[0, 0]], dtype="float32")
        with patch.object(rag, "get_model", return_value=model), patch.object(rag, "get_knowledge_base", return_value=base):
            self.assertEqual(len(rag.retrieve("q", top_k=4)), 1)
            self.assertEqual(rag.retrieve("q", scope="Uploaded PDFs only"), [])

    def test_upload_index_isolated_and_keeps_pages(self):
        reader = SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: "Water crops carefully.")])
        model = Mock()
        model.encode.return_value = np.array([[0, 0]], dtype="float32")
        with patch.object(rag, "PdfReader", return_value=reader), patch.object(rag, "get_model", return_value=model):
            indexed = rag.index_uploads([("notes.pdf", b"%PDF-test")])
            results = rag.retrieve("q", uploaded=indexed, scope="Uploaded PDFs only")
        self.assertEqual(results[0]["source"], "Uploaded: notes.pdf")
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(indexed["index"].ntotal, 1)

    def test_invalid_encrypted_and_scanned_pdfs(self):
        with self.assertRaisesRegex(ValueError, "not a PDF"):
            rag.index_uploads([("bad.pdf", b"not a PDF")])
        with patch.object(rag, "PdfReader", return_value=SimpleNamespace(is_encrypted=True)):
            with self.assertRaisesRegex(ValueError, "Password"):
                rag.index_uploads([("locked.pdf", b"%PDF-test")])
        reader = SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: "")])
        with patch.object(rag, "PdfReader", return_value=reader):
            with self.assertRaisesRegex(ValueError, "OCR"):
                rag.index_uploads([("scan.pdf", b"%PDF-test")])


if __name__ == "__main__":
    unittest.main()
