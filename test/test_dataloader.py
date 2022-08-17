from dataloader import load_mnist


def test_load_mnist():

    dataloader = load_mnist(batch_size=16)

    for batch in dataloader:
        print(batch[0].shape)
        print(batch[1].shape)

        # print()
        break

    assert batch.shape == (16, 28, 28)