{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "DataMiningTests.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMBYoDylic2zsKQF6qdSz+7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JesusCab/Data-Mining-Python/blob/main/DataMiningTests.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "**Mineria De Datos - Jesus Adrian Caballero Nagaya**\n",
        "\n",
        "1.- Adquisición de datos\n",
        "\n",
        "2.- Limpieza de datos.\n",
        "\n",
        "3.- Análisis de datos.\n",
        "\n",
        "4.- Graficación.\n",
        "\n",
        "5.- Prueba de hipótesis.\n",
        "\n",
        "6.- Regresión lineal\n",
        "\n",
        "7.- Forecasting\n",
        "\n",
        "8.-Clasificaion\n",
        "\n",
        "9.-Clustering\n"
      ],
      "metadata": {
        "id": "FYAD_p2y9kdk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **1.- Adquisición de datos**"
      ],
      "metadata": {
        "id": "KhnQ7q0lRdcQ"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "pSOwAQ060uRp",
        "outputId": "27f08e00-4ee7-45f8-c909-3d6a86e17528"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Unnamed: 0  mark   model generation_name  year  mileage  vol_engine  \\\n",
              "0             0  opel   combo      gen-d-2011  2015   139568        1248   \n",
              "1             1  opel   combo      gen-d-2011  2018    31991        1499   \n",
              "2             2  opel   combo      gen-d-2011  2015   278437        1598   \n",
              "3             3  opel   combo      gen-d-2011  2016    47600        1248   \n",
              "4             4  opel   combo      gen-d-2011  2014   103000        1400   \n",
              "..          ...   ...     ...             ...   ...      ...         ...   \n",
              "495         495  opel  antara             NaN  2008   295000        1991   \n",
              "496         496  opel  antara             NaN  2008   212000        1991   \n",
              "497         497  opel  antara             NaN  2009   220122        1991   \n",
              "498         498  opel  antara             NaN  2008   172000        2405   \n",
              "499         499  opel  antara             NaN  2007   236000        1991   \n",
              "\n",
              "         fuel                city       province  price  \n",
              "0      Diesel               Janki    Mazowieckie  35900  \n",
              "1      Diesel            Katowice        Śląskie  78501  \n",
              "2      Diesel               Brzeg       Opolskie  27000  \n",
              "3      Diesel           Korfantów       Opolskie  30800  \n",
              "4         CNG     Tarnowskie Góry        Śląskie  35900  \n",
              "..        ...                 ...            ...    ...  \n",
              "495    Diesel                Żory        Śląskie  22900  \n",
              "496    Diesel                Piła  Wielkopolskie  18900  \n",
              "497    Diesel              Kraków    Małopolskie  26900  \n",
              "498  Gasoline            Ostrówek        Łódzkie  29900  \n",
              "499    Diesel  Środa Wielkopolska  Wielkopolskie  22900  \n",
              "\n",
              "[500 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0736025d-02b0-4293-9287-5350b17c76f5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>mark</th>\n",
              "      <th>model</th>\n",
              "      <th>generation_name</th>\n",
              "      <th>year</th>\n",
              "      <th>mileage</th>\n",
              "      <th>vol_engine</th>\n",
              "      <th>fuel</th>\n",
              "      <th>city</th>\n",
              "      <th>province</th>\n",
              "      <th>price</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>opel</td>\n",
              "      <td>combo</td>\n",
              "      <td>gen-d-2011</td>\n",
              "      <td>2015</td>\n",
              "      <td>139568</td>\n",
              "      <td>1248</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Janki</td>\n",
              "      <td>Mazowieckie</td>\n",
              "      <td>35900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>opel</td>\n",
              "      <td>combo</td>\n",
              "      <td>gen-d-2011</td>\n",
              "      <td>2018</td>\n",
              "      <td>31991</td>\n",
              "      <td>1499</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Katowice</td>\n",
              "      <td>Śląskie</td>\n",
              "      <td>78501</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>opel</td>\n",
              "      <td>combo</td>\n",
              "      <td>gen-d-2011</td>\n",
              "      <td>2015</td>\n",
              "      <td>278437</td>\n",
              "      <td>1598</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Brzeg</td>\n",
              "      <td>Opolskie</td>\n",
              "      <td>27000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>opel</td>\n",
              "      <td>combo</td>\n",
              "      <td>gen-d-2011</td>\n",
              "      <td>2016</td>\n",
              "      <td>47600</td>\n",
              "      <td>1248</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Korfantów</td>\n",
              "      <td>Opolskie</td>\n",
              "      <td>30800</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>opel</td>\n",
              "      <td>combo</td>\n",
              "      <td>gen-d-2011</td>\n",
              "      <td>2014</td>\n",
              "      <td>103000</td>\n",
              "      <td>1400</td>\n",
              "      <td>CNG</td>\n",
              "      <td>Tarnowskie Góry</td>\n",
              "      <td>Śląskie</td>\n",
              "      <td>35900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>495</th>\n",
              "      <td>495</td>\n",
              "      <td>opel</td>\n",
              "      <td>antara</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2008</td>\n",
              "      <td>295000</td>\n",
              "      <td>1991</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Żory</td>\n",
              "      <td>Śląskie</td>\n",
              "      <td>22900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>496</th>\n",
              "      <td>496</td>\n",
              "      <td>opel</td>\n",
              "      <td>antara</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2008</td>\n",
              "      <td>212000</td>\n",
              "      <td>1991</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Piła</td>\n",
              "      <td>Wielkopolskie</td>\n",
              "      <td>18900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>497</th>\n",
              "      <td>497</td>\n",
              "      <td>opel</td>\n",
              "      <td>antara</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2009</td>\n",
              "      <td>220122</td>\n",
              "      <td>1991</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Kraków</td>\n",
              "      <td>Małopolskie</td>\n",
              "      <td>26900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>498</th>\n",
              "      <td>498</td>\n",
              "      <td>opel</td>\n",
              "      <td>antara</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2008</td>\n",
              "      <td>172000</td>\n",
              "      <td>2405</td>\n",
              "      <td>Gasoline</td>\n",
              "      <td>Ostrówek</td>\n",
              "      <td>Łódzkie</td>\n",
              "      <td>29900</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>499</th>\n",
              "      <td>499</td>\n",
              "      <td>opel</td>\n",
              "      <td>antara</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2007</td>\n",
              "      <td>236000</td>\n",
              "      <td>1991</td>\n",
              "      <td>Diesel</td>\n",
              "      <td>Środa Wielkopolska</td>\n",
              "      <td>Wielkopolskie</td>\n",
              "      <td>22900</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>500 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0736025d-02b0-4293-9287-5350b17c76f5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-0736025d-02b0-4293-9287-5350b17c76f5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-0736025d-02b0-4293-9287-5350b17c76f5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ],
      "source": [
        "import requests\n",
        "import io\n",
        "import numpy as np\n",
        "from bs4 import BeautifulSoup\n",
        "import pandas as pd\n",
        "from tabulate import tabulate\n",
        "from typing import Tuple, List\n",
        "from scipy import stats\n",
        "import re\n",
        "from datetime import datetime\n",
        "import matplotlib.pyplot as plt\n",
        "import plotly.graph_objects as go\n",
        "from sklearn.linear_model import LinearRegression\n",
        "import statsmodels.formula.api as smf\n",
        "\n",
        "def get_soup(url: str) -> BeautifulSoup:\n",
        "    response = requests.get(url)\n",
        "    return BeautifulSoup(response.content, 'html.parser')\n",
        "\n",
        "def get_csv_from_url(url:str) -> pd.DataFrame:\n",
        "    s=requests.get(url).content\n",
        "    return pd.read_csv(io.StringIO(s.decode('utf-8')))\n",
        "\n",
        "def print_tabulate(df: pd.DataFrame):\n",
        "    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))\n",
        "\n",
        "\n",
        "df = get_csv_from_url(\"https://raw.githubusercontent.com/JesusCab/Data-Mining-Python/main/Car_Prices_Poland_Kaggle.csv\")\n",
        "df.head(500)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **2.- Limpieza de datos.**"
      ],
      "metadata": {
        "id": "LfwC_W4-RhSo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ctvhlgYUH1Oj",
        "outputId": "f90e847e-aa3c-4f8f-9b72-074c513cf2f4"
      },
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(117927, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isna().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VffDAtNNIKC9",
        "outputId": "29659bb2-5c0d-4446-e4be-a7214e8e35b2"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Unnamed: 0             0\n",
              "mark                   0\n",
              "model                  0\n",
              "generation_name    30085\n",
              "year                   0\n",
              "mileage                0\n",
              "vol_engine             0\n",
              "fuel                   0\n",
              "city                   0\n",
              "province               0\n",
              "price                  0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = df.dropna()"
      ],
      "metadata": {
        "id": "SCLnOvm8IZAS"
      },
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SKQ-WC35Iexx",
        "outputId": "6258dca9-5cc9-46db-cc47-86c6f19278e2"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(87842, 11)"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "price_table = df[\"price\"]\n",
        "price_table"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B5yan3M091Tr",
        "outputId": "b5f84732-294d-49c5-82f8-708c3361bfc7"
      },
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0          35900\n",
              "1          78501\n",
              "2          27000\n",
              "3          30800\n",
              "4          35900\n",
              "           ...  \n",
              "117922    222790\n",
              "117923    229900\n",
              "117924    135000\n",
              "117925    154500\n",
              "117926    130000\n",
              "Name: price, Length: 87842, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model_frame = pd.DataFrame(data=df['model'])\n",
        "model_frame"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "94GRnugh94-P",
        "outputId": "25b61c75-51ad-4227-e224-a2a18f9ec4d1"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        model\n",
              "0       combo\n",
              "1       combo\n",
              "2       combo\n",
              "3       combo\n",
              "4       combo\n",
              "...       ...\n",
              "117922  xc-90\n",
              "117923  xc-90\n",
              "117924  xc-90\n",
              "117925  xc-90\n",
              "117926  xc-90\n",
              "\n",
              "[87842 rows x 1 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9a63d42e-7c63-4815-873b-09ceedbaec90\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>model</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>combo</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>combo</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>combo</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>combo</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>combo</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117922</th>\n",
              "      <td>xc-90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117923</th>\n",
              "      <td>xc-90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117924</th>\n",
              "      <td>xc-90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117925</th>\n",
              "      <td>xc-90</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117926</th>\n",
              "      <td>xc-90</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>87842 rows × 1 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9a63d42e-7c63-4815-873b-09ceedbaec90')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9a63d42e-7c63-4815-873b-09ceedbaec90 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9a63d42e-7c63-4815-873b-09ceedbaec90');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 56
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "type(df)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Pi_Mq0FAZ3FA",
        "outputId": "6cd40df4-cfff-4757-f15f-0ba8ada47c2c"
      },
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "pandas.core.frame.DataFrame"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **3.- Análisis de datos.**"
      ],
      "metadata": {
        "id": "VbKNXaWxRxDf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.mark.unique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vH96C8WnaPmy",
        "outputId": "ca51d273-3ee7-4abb-cdfa-a6893118ef06"
      },
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['opel', 'audi', 'bmw', 'volkswagen', 'ford', 'mercedes-benz',\n",
              "       'renault', 'toyota', 'skoda', 'citroen', 'fiat', 'honda',\n",
              "       'hyundai', 'kia', 'mazda', 'mitsubishi', 'nissan', 'peugeot',\n",
              "       'seat', 'volvo'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 58
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.model.unique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t0YStJtraYiA",
        "outputId": "dc4d1508-859c-4fd6-cdfb-784dc654f300"
      },
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['combo', 'vectra', 'agila', 'astra', 'corsa', 'frontera',\n",
              "       'insignia', 'zafira', 'a3', 'meriva', 'omega', 'tigra', '80', 'a4',\n",
              "       'a5', 'a6', 'a6-allroad', 'a7', 'a8', 'q5', 'q7', 's3', 's8', 'tt',\n",
              "       'seria-1', 'seria-3', 'seria-5', 'seria-6', 'seria-7', 'seria-8',\n",
              "       'x1', 'x3', 'x4', 'x5', 'x6', 'caddy', 'golf', 'jetta', 'passat',\n",
              "       'polo', 'sharan', 'tiguan', 'touareg', 'touran', 'transporter',\n",
              "       'c-max', 'fiesta', 'focus', 'galaxy', 'ka', 'kuga', 'mondeo',\n",
              "       's-max', 'transit', 'transit-connect', 'cl-klasa', 'clk-klasa',\n",
              "       'cls-klasa', 'gl-klasa', 'gle-klasa', 'a-klasa', 'b-klasa',\n",
              "       'c-klasa', 'e-klasa', 'g-klasa', 's-klasa', 'm-klasa', 'sl',\n",
              "       'sprinter', 'vito', 'clio', 'espace', 'grand-scenic', 'kangoo',\n",
              "       'laguna', 'megane', 'scenic', 'trafic', 'twingo', 'auris',\n",
              "       'avensis', 'aygo', 'corolla', 'land-cruiser', 'prius', 'rav4',\n",
              "       'yaris', 'fabia', 'octavia', 'superb', 'berlingo', 'c4-picasso',\n",
              "       'c5', 'bravo', 'doblo', 'panda', 'punto', 'tipo', 'accord', 'cr-v',\n",
              "       'hr-v', 'jazz', 'civic', 'elantra', 'i10', 'i20', 'i30',\n",
              "       'santa-fe', 'tucson', 'carens', 'ceed', 'picanto', 'sorento',\n",
              "       'soul', 'sportage', '2', '3', '5', '6', 'colt', 'lancer',\n",
              "       'outlander', 'space-star', 'almera', 'juke', 'micra', 'note',\n",
              "       'patrol', 'primera', 'qashqai', 'x-trail', '3008', '5008',\n",
              "       'partner', 'alhambra', 'ibiza', 'leon', 'toledo', 's40', 's60',\n",
              "       's80', 'v40', 'v60', 'v70', 'xc-60', 'xc-90'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.price.unique"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ol4zWeHXagZb",
        "outputId": "49ce86be-6e26-48d4-a557-b3830423076b"
      },
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<bound method Series.unique of 0          35900\n",
              "1          78501\n",
              "2          27000\n",
              "3          30800\n",
              "4          35900\n",
              "           ...  \n",
              "117922    222790\n",
              "117923    229900\n",
              "117924    135000\n",
              "117925    154500\n",
              "117926    130000\n",
              "Name: price, Length: 87842, dtype: int64>"
            ]
          },
          "metadata": {},
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "gender_frame = pd.DataFrame(data=df['price'])\n",
        "gender_frame \n",
        "x=max(price_table)\n",
        "y=min(price_table)\n",
        "z= df[\"price\"].mean()\n",
        "w=df[\"price\"].median()\n",
        "m = df[\"price\"].mode()\n",
        "r=df[\"price\"].count()\n",
        "s=df[\"price\"].sum()\n",
        "k=df[\"price\"].kurtosis()\n",
        "v=df[\"price\"].var()\n",
        "d=df[\"price\"].std()\n",
        "print(\"Max {0}\\n Min {1}\\n Promedio {2}\\n Mediana {3}\\n moda {4}\\n Conteo {5}\\n Suma {6}\\n Kutosis {7}\\n Varianza {8}\\n Desviacion Estandar {9}\\n\".format(x,y,z,w,m,r,s,k,v,d))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1YTKXl-p96In",
        "outputId": "fc33888a-f413-4995-d9b8-7646a57a38ce"
      },
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Max 2399900\n",
            " Min 900\n",
            " Promedio 63744.12083058218\n",
            " Mediana 37900.0\n",
            " moda 0    19900\n",
            "dtype: int64\n",
            " Conteo 87842\n",
            " Suma 5599411062\n",
            " Kutosis 30.920629610578416\n",
            " Varianza 5894975509.403856\n",
            " Desviacion Estandar 76778.74386445676\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "g = df.groupby(['model','mark']).mean()\n",
        "g"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 454
        },
        "id": "y_SyLp819-kS",
        "outputId": "0b6f40e5-de08-4533-a871-499d5134b75c"
      },
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                Unnamed: 0         year        mileage   vol_engine  \\\n",
              "model  mark                                                           \n",
              "2      mazda     97622.500  2011.075893  128518.066964  1411.049107   \n",
              "3      mazda     98054.500  2013.059375  118840.732813  1858.639062   \n",
              "3008   peugeot  109782.500  2015.152961  123283.000000  1587.442434   \n",
              "5      mazda     98502.500  2008.414062  198396.289062  1912.843750   \n",
              "5008   peugeot  110278.500  2013.752604  160301.049479  1710.984375   \n",
              "...                    ...          ...            ...          ...   \n",
              "x6     bmw       34838.500  2015.075000  124347.868750  3115.237500   \n",
              "xc-60  volvo    116422.500  2016.440104  120865.558594  2103.890625   \n",
              "xc-90  volvo    117414.500  2016.705078  103758.671875  2074.255859   \n",
              "yaris  toyota    74086.500  2013.528125  109041.981250  1272.831250   \n",
              "zafira opel      10468.625  2008.911719  199256.878906  1789.228125   \n",
              "\n",
              "                        price  \n",
              "model  mark                    \n",
              "2      mazda     25803.669643  \n",
              "3      mazda     47398.850000  \n",
              "3008   peugeot   76473.597039  \n",
              "5      mazda     18826.378906  \n",
              "5008   peugeot   56839.205729  \n",
              "...                       ...  \n",
              "x6     bmw      213879.168750  \n",
              "xc-60  volvo    135416.653646  \n",
              "xc-90  volvo    201558.773438  \n",
              "yaris  toyota    38927.764583  \n",
              "zafira opel      22751.250000  \n",
              "\n",
              "[146 rows x 5 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9ec6f1ac-122c-4b10-81d6-b513bd5c3c4d\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>year</th>\n",
              "      <th>mileage</th>\n",
              "      <th>vol_engine</th>\n",
              "      <th>price</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>model</th>\n",
              "      <th>mark</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <th>mazda</th>\n",
              "      <td>97622.500</td>\n",
              "      <td>2011.075893</td>\n",
              "      <td>128518.066964</td>\n",
              "      <td>1411.049107</td>\n",
              "      <td>25803.669643</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <th>mazda</th>\n",
              "      <td>98054.500</td>\n",
              "      <td>2013.059375</td>\n",
              "      <td>118840.732813</td>\n",
              "      <td>1858.639062</td>\n",
              "      <td>47398.850000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3008</th>\n",
              "      <th>peugeot</th>\n",
              "      <td>109782.500</td>\n",
              "      <td>2015.152961</td>\n",
              "      <td>123283.000000</td>\n",
              "      <td>1587.442434</td>\n",
              "      <td>76473.597039</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <th>mazda</th>\n",
              "      <td>98502.500</td>\n",
              "      <td>2008.414062</td>\n",
              "      <td>198396.289062</td>\n",
              "      <td>1912.843750</td>\n",
              "      <td>18826.378906</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5008</th>\n",
              "      <th>peugeot</th>\n",
              "      <td>110278.500</td>\n",
              "      <td>2013.752604</td>\n",
              "      <td>160301.049479</td>\n",
              "      <td>1710.984375</td>\n",
              "      <td>56839.205729</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>x6</th>\n",
              "      <th>bmw</th>\n",
              "      <td>34838.500</td>\n",
              "      <td>2015.075000</td>\n",
              "      <td>124347.868750</td>\n",
              "      <td>3115.237500</td>\n",
              "      <td>213879.168750</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>xc-60</th>\n",
              "      <th>volvo</th>\n",
              "      <td>116422.500</td>\n",
              "      <td>2016.440104</td>\n",
              "      <td>120865.558594</td>\n",
              "      <td>2103.890625</td>\n",
              "      <td>135416.653646</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>xc-90</th>\n",
              "      <th>volvo</th>\n",
              "      <td>117414.500</td>\n",
              "      <td>2016.705078</td>\n",
              "      <td>103758.671875</td>\n",
              "      <td>2074.255859</td>\n",
              "      <td>201558.773438</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>yaris</th>\n",
              "      <th>toyota</th>\n",
              "      <td>74086.500</td>\n",
              "      <td>2013.528125</td>\n",
              "      <td>109041.981250</td>\n",
              "      <td>1272.831250</td>\n",
              "      <td>38927.764583</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>zafira</th>\n",
              "      <th>opel</th>\n",
              "      <td>10468.625</td>\n",
              "      <td>2008.911719</td>\n",
              "      <td>199256.878906</td>\n",
              "      <td>1789.228125</td>\n",
              "      <td>22751.250000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>146 rows × 5 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9ec6f1ac-122c-4b10-81d6-b513bd5c3c4d')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9ec6f1ac-122c-4b10-81d6-b513bd5c3c4d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9ec6f1ac-122c-4b10-81d6-b513bd5c3c4d');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe()\n",
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B0CHa7Oj-Ar3",
        "outputId": "b1e7a843-c468-4bdd-e43d-565f5f914eef"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 87842 entries, 0 to 117926\n",
            "Data columns (total 11 columns):\n",
            " #   Column           Non-Null Count  Dtype \n",
            "---  ------           --------------  ----- \n",
            " 0   Unnamed: 0       87842 non-null  int64 \n",
            " 1   mark             87842 non-null  object\n",
            " 2   model            87842 non-null  object\n",
            " 3   generation_name  87842 non-null  object\n",
            " 4   year             87842 non-null  int64 \n",
            " 5   mileage          87842 non-null  int64 \n",
            " 6   vol_engine       87842 non-null  int64 \n",
            " 7   fuel             87842 non-null  object\n",
            " 8   city             87842 non-null  object\n",
            " 9   province         87842 non-null  object\n",
            " 10  price            87842 non-null  int64 \n",
            "dtypes: int64(5), object(6)\n",
            "memory usage: 8.0+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **4.- Graficación.**"
      ],
      "metadata": {
        "id": "HTxla5iSR5lW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(df['year'], df['price'], label='price')\n",
        "plt.plot(df['year'], df['model'], label='model')\n",
        "\n",
        "plt.xlabel('Fecha')\n",
        "plt.ylabel('Precio')\n",
        "plt.title('Popularidad de términos relacionados con IA')\n",
        "plt.grid(True)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 294
        },
        "id": "fhhieZuu-HEq",
        "outputId": "4f3acf52-5083-4834-e9a5-aa528c853163"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "m = df['model']\n",
        "m\n"
      ],
      "metadata": {
        "id": "MlCYAd-n0aqc",
        "outputId": "dc2c49b2-5144-4ced-9db8-dd1d7b92dfea",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0         combo\n",
              "1         combo\n",
              "2         combo\n",
              "3         combo\n",
              "4         combo\n",
              "          ...  \n",
              "117922    xc-90\n",
              "117923    xc-90\n",
              "117924    xc-90\n",
              "117925    xc-90\n",
              "117926    xc-90\n",
              "Name: model, Length: 87842, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.bar(df['year'], df['price'], width=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cmnXzTw71A84",
        "outputId": "756dde01-ea0f-4500-b325-c4d8678c73a2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BarContainer object of 87842 artists>"
            ]
          },
          "metadata": {},
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig = go.Figure()\n",
        "\n",
        "fig.add_trace(go.Bar(x=df['model'], y=df['price']))\n",
        "\n",
        "fig.update_layout(\n",
        "    xaxis=dict(\n",
        "        title_text='model',\n",
        "        titlefont=dict(size=30),\n",
        "    ),\n",
        "    yaxis=dict(\n",
        "        title_text=\"price\",\n",
        "        \n",
        "        titlefont=dict(size=30)\n",
        "    ),\n",
        "    title=\"acumulado de dinero gastado por modelo de auto\"\n",
        ")\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "id": "yZRHCmy30zHb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = go.Figure()\n",
        "\n",
        "fig.add_trace(go.Bar(x=df['city'], y=df['price']))\n",
        "\n",
        "fig.update_layout(\n",
        "    xaxis=dict(\n",
        "        title_text='city',\n",
        "        titlefont=dict(size=30),\n",
        "    ),\n",
        "    yaxis=dict(\n",
        "        title_text=\"price\",\n",
        "        \n",
        "        titlefont=dict(size=30)\n",
        "    ),\n",
        "    title=\"acumulado de dinero gastado por ciudad de polonia en autos\"\n",
        ")\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "id": "f4O_YSWuirf3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **5.- Prueba de Hipotesis**"
      ],
      "metadata": {
        "id": "MCJrDkkbTl8G"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "PrecioAudi = df.price[df.mark ==\"audi\"]\n",
        "PrecioMercedes = df.price[df.mark =='mercedes-benz']\n",
        "PrecioBmw = df.price[df.mark =='bmw']\n"
      ],
      "metadata": {
        "id": "Uc2a9sDPiCZP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from scipy import stats\n",
        "anova = stats.f_oneway(PrecioAudi, PrecioMercedes, PrecioBmw)\n",
        "anova"
      ],
      "metadata": {
        "id": "dfKnCGW0QjcO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.mark.unique()"
      ],
      "metadata": {
        "id": "lvC0UJ7Ca6_f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stats.ttest_ind(df.price.loc[df.mark=='audi'],\n",
        "                df.price.loc[df.mark=='volkswagen'],nan_policy='omit')"
      ],
      "metadata": {
        "id": "HFrjPslWZwu8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = go.Figure()\n",
        "# Hipotesis de que los carros de Audi son mas costosos que los de volkswagen\n",
        "fig.add_trace(go.Box(y=df.price.loc[df.mark=='audi'],name=\"AUDI\"))\n",
        "fig.add_trace(go.Box(y=df.price.loc[df.mark=='volkswagen'],name=\"VOLKSWAGEN\"))\n",
        "fig.update_layout(\n",
        "    title={\n",
        "        'text': \"Precios de carros\",\n",
        "        'y':0.8,\n",
        "        'x':0.4,\n",
        "        'xanchor': 'center',\n",
        "        'yanchor': 'top'})\n",
        "fig.show()"
      ],
      "metadata": {
        "id": "qgxvJyiQbSXE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **6.- Regresión lineal**"
      ],
      "metadata": {
        "id": "B81Q3fbMSGDL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['price']= df['price']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['price'])\n",
        "\n",
        "meanLine = [[df['price'].mean()],[df['price'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['price'])\n",
        "plt.title(\"Regresion Lineal año-precio\")"
      ],
      "metadata": {
        "id": "Xa76eL5vkqJm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['price']= df['price']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['price'])\n",
        "print(\"b = \", modelo.intercept_)\n",
        "print(\"m = \", modelo.coef_)\n",
        "\n",
        "print(modelo.predict([[0],[50]]))\n",
        "print(df['price'].mean())\n",
        "\n",
        "meanLine = [[df['price'].mean()],[df['price'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['price'])\n",
        "plt.plot([[1940],[2022]], modelo.predict([[0],[50]]), color = 'red')\n",
        "plt.plot([[1940],[2022]], meanLine, color = 'green')\n",
        "\n",
        "plt.title(\"Regresion Lineal año-precio\")\n",
        "\n"
      ],
      "metadata": {
        "id": "kPiics5LTbg_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['mileage']= df['mileage']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "#print(datePrice)\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['mileage'])\n",
        "print(\"b = \", modelo.intercept_)\n",
        "print(\"m = \", modelo.coef_)\n",
        "\n",
        "print(modelo.predict([[0],[50]]))\n",
        "print(df['mileage'].mean())\n",
        "meanLine = [[df['mileage'].mean()],[df['mileage'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['mileage'])\n",
        "plt.plot([[1940],[2022]], modelo.predict([[0],[50]]), color = 'red')\n",
        "plt.plot([[1940],[2022]], meanLine, color = 'green')\n",
        "\n",
        "plt.title(\"Regresion Lineal año-millage\")\n"
      ],
      "metadata": {
        "id": "Hphacav9wj7x"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['vol_engine']= df['vol_engine']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['vol_engine'])\n",
        "print(\"b = \", modelo.intercept_)\n",
        "print(\"m = \", modelo.coef_)\n",
        "\n",
        "print(modelo.predict([[0],[50]]))\n",
        "print(df['vol_engine'].mean())\n",
        "\n",
        "meanLine = [[df['vol_engine'].mean()],[df['vol_engine'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['vol_engine'])\n",
        "plt.plot([[1940],[2022]], modelo.predict([[0],[50]]), color = 'red')\n",
        "plt.plot([[1940],[2022]], meanLine, color = 'green')\n",
        "plt.title(\"Regresion Lineal año-Volumen del Motor\")\n"
      ],
      "metadata": {
        "id": "LejWHVjmUV9k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **7.- Forecasting**"
      ],
      "metadata": {
        "id": "fXyV2dHFSZ76"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['price']= df['price']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "results = smf.ols('price~index', datePrice).fit()\n",
        "predicts = results.predict()\n",
        "\n",
        "bands = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]\n",
        "coef = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']\n",
        "m = coef.values[1]\n",
        "b = coef.values[0]\n",
        "low = bands['[0.025'][0]\n",
        "hi = bands['0.975]'][0]\n",
        "\n",
        "\n",
        "\n",
        "lowBand = m * datePrice['index'] + low\n",
        "highBand = m * datePrice['index'] + hi\n",
        "\n",
        "\n",
        "b0 = results.params[0]\n",
        "b1 = results.params[1]\n",
        "datePrice['prediction'] = b0 + b1*datePrice['price']\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['price'])\n",
        "print(\"b = \", modelo.intercept_)\n",
        "print(\"m = \", modelo.coef_)\n",
        "\n",
        "print(modelo.predict([[0],[50]]))\n",
        "print(df['price'].mean())\n",
        "\n",
        "meanLine = [[df['price'].mean()],[df['price'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['price'])\n",
        "plt.plot([[1940],[2022]], modelo.predict([[0],[50]]), color = 'red')\n",
        "plt.plot([[1940],[2022]], meanLine, color = 'green')\n",
        "\n",
        "plt.plot(datePrice['year'], lowBand, color='orange')\n",
        "plt.plot(datePrice['year'], highBand, color='orange')\n",
        "plt.fill_between(datePrice['year'], lowBand, highBand, alpha=0.4, color = 'purple')\n",
        "plt.title(\"Regresion Lineal con forecasting año-precio\")\n"
      ],
      "metadata": {
        "id": "MwOsfhd4bBlx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "datePrice = pd.DataFrame()\n",
        "datePrice['year'] = df['year']\n",
        "datePrice ['mileage']= df['mileage']\n",
        "\n",
        "datePrice = datePrice.groupby(pd.Grouper(key='year')).mean().reset_index()\n",
        "datePrice=datePrice.reset_index()\n",
        "\n",
        "\n",
        "results = smf.ols('mileage~index', datePrice).fit()\n",
        "predicts = results.predict()\n",
        "\n",
        "bands = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]\n",
        "coef = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']\n",
        "m = coef.values[1]\n",
        "b = coef.values[0]\n",
        "low = bands['[0.025'][0]\n",
        "hi = bands['0.975]'][0]\n",
        "\n",
        "\n",
        "lowBand = m * datePrice['index'] + low\n",
        "highBand = m * datePrice['index'] + hi\n",
        "\n",
        "\n",
        "b0 = results.params[0]\n",
        "b1 = results.params[1]\n",
        "datePrice['prediction'] = b0 + b1*datePrice['mileage']\n",
        "\n",
        "modelo = LinearRegression().fit(datePrice['index'].values.reshape((-1,1)), datePrice['mileage'])\n",
        "print(\"b = \", modelo.intercept_)\n",
        "print(\"m = \", modelo.coef_)\n",
        "\n",
        "print(modelo.predict([[0],[50]]))\n",
        "print(df['mileage'].mean())\n",
        "\n",
        "meanLine = [[df['mileage'].mean()],[df['mileage'].mean()]]\n",
        "\n",
        "plt.scatter(datePrice['year'], datePrice['mileage'])\n",
        "plt.plot([[1940],[2022]], modelo.predict([[0],[50]]), color = 'red')\n",
        "plt.plot([[1940],[2022]], meanLine, color = 'green')\n",
        "\n",
        "plt.plot(datePrice['year'], lowBand, color='orange')\n",
        "plt.plot(datePrice['year'], highBand, color='orange')\n",
        "plt.fill_between(datePrice['year'], lowBand, highBand, alpha=0.4, color = 'purple')"
      ],
      "metadata": {
        "id": "oeZ8Osq9hROO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **8.-Clasificacion**"
      ],
      "metadata": {
        "id": "Z2nww7fxSsrm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "H467xv2-SwcX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **9.-Clustering**"
      ],
      "metadata": {
        "id": "JjMSIfTeSw3v"
      }
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "ujS5ww4PS0kE"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}