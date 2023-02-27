from ml_swissknife import crfm_utils, openai_utils


def assert_strict_type_match(obj1, obj2, recursive=True, ignore_key_set_mismatch=True, ignore_length_mismatch=True):
    assert type(obj1) == type(obj2)
    if recursive:
        if isinstance(obj1, dict):
            if not ignore_key_set_mismatch:
                assert set(obj1.keys()) == set(obj2.keys())
            for k in obj1.keys():
                if k in obj2:
                    assert_strict_type_match(obj1[k], obj2[k], recursive=recursive,
                                             ignore_key_set_mismatch=ignore_key_set_mismatch)
        if isinstance(obj1, (list, tuple)):
            if not ignore_length_mismatch:
                assert len(obj1) == len(obj2)
            for x, y in zip(obj1, obj2):
                assert_strict_type_match(x, y, recursive=recursive, ignore_key_set_mismatch=ignore_key_set_mismatch)


def test_crfm_completion():
    """Test the return format between the two APIs are the same when crfm_completion is forced to do so."""
    prompts = ["Life is like a box of", "Life is"]

    # Note the types don't match when logprobs=None. CRFM API always gives logprobs, whereas OpenAI API doesn't.
    single_sample_decoding_args = openai_utils.OpenAIDecodingArguments(logprobs=3)
    multi_sample_decoding_args = openai_utils.OpenAIDecodingArguments(n=2, logprobs=3)

    for decoding_args in (single_sample_decoding_args, multi_sample_decoding_args):
        for return_text in (True, False,):
            crfm_ret = crfm_utils.crfm_completion(
                prompts, return_text=return_text, decoding_args=decoding_args, return_openai_object=True)
            openai_ret = openai_utils.openai_completion(
                prompts, return_text=return_text, decoding_args=decoding_args, batch_size=2)
            assert_strict_type_match(
                crfm_ret, openai_ret, recursive=True, ignore_key_set_mismatch=True, ignore_length_mismatch=True)
