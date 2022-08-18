from dataloader import load_mnist


def test_load_mnist():

    dataloader = load_mnist(batch_size=16)

    for (imgs, _) in dataloader:
        break

    assert imgs.shape == (16, 1, 28, 28)