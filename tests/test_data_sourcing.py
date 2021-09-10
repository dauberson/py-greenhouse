import pandera as pa
from src import data_sourcing


def test_data_sourcing_get():

    df = data_sourcing.get_example()

    cats = ["Positivo", "Neutro", "Negativo"]

    schema = pa.DataFrameSchema(
        {
            "cats": pa.Column(str, checks=pa.Check.isin(cats), nullable=True),
            "text": pa.Column(str, nullable=True),
        }
    )

    schema(df)
