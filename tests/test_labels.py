from labels import ID2LABEL, LABEL2ID, LABEL_DESCRIPTIONS, LABELS, NUM_LABELS, TECHNIQUE_TO_LABEL


def test_num_labels_matches_list_length():
    assert NUM_LABELS == len(LABELS)


def test_label2id_and_id2label_are_inverses():
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label
    assert len(LABEL2ID) == len(LABELS)


def test_every_label_has_a_description():
    assert set(LABEL_DESCRIPTIONS) == set(LABELS)


def test_technique_mapping_only_targets_known_labels():
    assert set(TECHNIQUE_TO_LABEL.values()) <= set(LABELS)
