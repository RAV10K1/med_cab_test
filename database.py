from pymongo import MongoClient
from os import getenv
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


__all__ = ('DataBase',)


class DataBase:

    def connect_db(self):
        """ MongoDB Table Connection """
        return MongoClient(
            f"mongodb+srv://{getenv('MONGODB_USER')}:{getenv('MONGODB_PASS')}"
            f"@{getenv('MONGODB_URI')}/test?retryWrites=true&w=majority"
        ).med_cab.strains

    def read_csv(self):
        return pd.read_csv('test_app/lemmatized_strains.csv')

    def make_db(self):
        """Creates and populates database in MongoDB"""
        db = self.connect_db()
        data = self.read_csv().to_dict(orient='records')
        for strain in data:
            strain['Effects'] = strain['Effects'].split(',')
            strain['Flavors'] = strain['Flavors'].split(',')
            strain['Cos Sim Strains'] = [
                data[int(idx)]['Strain'] for idx in strain['Cos Sim Strains'].split(',')
            ]
        db.insert_many(data)

    def find(self, query: dict) -> dict:
        return self.connect_db().find_one(query)


if __name__ == '__main__':
    data_model = DataBase()
    # data_model.make_db()
    data_model.connect_db()