{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Adarsh-hello/deepfake_detection/blob/main/deepfake_app11.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rQjDBTamsSPZ",
        "outputId": "5b0bb02c-d449-4ad1-d491-6c41c8fcab04"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.11/dist-packages (1.45.1)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.11/dist-packages (7.2.8)\n",
            "Requirement already satisfied: gdown in /usr/local/lib/python3.11/dist-packages (5.2.0)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.1.8)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.2.1)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.29.4)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (9.1.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.13.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.44)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.11/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.11/dist-packages (from gdown) (4.13.4)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from gdown) (3.18.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from gdown) (4.67.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (1.38.2)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2025.4.26)\n",
            "Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4->gdown) (2.7)\n",
            "Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.11/dist-packages (from requests[socks]->gdown) (1.7.1)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2025.4.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.24.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit pyngrok gdown\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gWMn1_UWj4Sw",
        "outputId": "9ac14a9c-f71b-4112-e7ea-dc6a11662402"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import os\n",
        "\n",
        "\n",
        "try:\n",
        "    import gdown\n",
        "except ImportError:\n",
        "    os.system('pip install gdown')\n",
        "    import gdown\n",
        "\n",
        "from tensorflow.keras.models import load_model\n",
        "\n",
        "\n",
        "st.set_page_config(page_title=\"Deepfake Detection using CNN\", page_icon=\"🧠\", layout=\"centered\")\n",
        "st.title(\"🧠 Deepfake Detection - CNN Model (with Google Drive)\")\n",
        "\n",
        "\n",
        "MODEL_PATH = \"Cnn_model.h5\"\n",
        "DRIVE_FILE_ID = \"1glEUFHmDxCFHWSRfPFS0BWxdk9jEMpJG\"\n",
        "\n",
        "\n",
        "def download_model():\n",
        "    if not os.path.exists(MODEL_PATH):\n",
        "        url = f\"https://drive.google.com/uc?id={DRIVE_FILE_ID}\"\n",
        "        gdown.download(url, MODEL_PATH, quiet=False)\n",
        "\n",
        "\n",
        "uploaded_file = st.file_uploader(\"📤 Upload an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
        "\n",
        "\n",
        "@st.cache_resource\n",
        "def load_cnn_model():\n",
        "    download_model()\n",
        "    cnn = load_model(MODEL_PATH)\n",
        "    return cnn\n",
        "\n",
        "try:\n",
        "    cnn_model = load_cnn_model()\n",
        "except Exception as e:\n",
        "    st.error(f\"❌ Error loading model: {e}\")\n",
        "    st.stop()\n",
        "\n",
        "\n",
        "labels = ['Real', 'Fake']\n",
        "\n",
        "\n",
        "def preprocess(img):\n",
        "    img = img.resize((128, 128))\n",
        "    img_array = np.array(img) / 255.0\n",
        "    if img_array.shape[-1] == 4:\n",
        "        img_array = img_array[..., :3]\n",
        "    return np.expand_dims(img_array, axis=0)\n",
        "\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    st.image(uploaded_file, caption=\"🖼 Uploaded Image\", use_column_width=False, width=250)\n",
        "    try:\n",
        "        image = Image.open(uploaded_file)\n",
        "        processed = preprocess(image)\n",
        "        cnn_pred = cnn_model.predict(processed)[0]\n",
        "        cnn_result = labels[np.argmax(cnn_pred)]\n",
        "\n",
        "        st.markdown(\"### 🔍 Prediction Result:\")\n",
        "        st.success(f\"🧠 CNN Prediction: {cnn_result} ({cnn_pred[np.argmax(cnn_pred)]*100:.2f}%)\")\n",
        "\n",
        "    except Exception as e:\n",
        "        st.error(f\"Prediction error: {e}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pkill -f ngrok\n"
      ],
      "metadata": {
        "id": "8IHfTdT8mm2_"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "-ETcsDlHdkK3"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n"
      ],
      "metadata": {
        "id": "tGJGp1KGdj_T"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YNjmPtyMhrjA",
        "outputId": "30fdf834-54af-4a17-b1b4-891e21b69479"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://104.197.89.31:8501\u001b[0m\n",
            "\u001b[0m\n",
            "NgrokTunnel: \"https://635d-104-197-89-31.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ],
      "source": [
        "from pyngrok import ngrok\n",
        "import time\n",
        "import threading\n",
        "\n",
        "\n",
        "def start_streamlit():\n",
        "    !streamlit run app.py\n",
        "\n",
        "thread = threading.Thread(target=start_streamlit)\n",
        "thread.start()\n",
        "\n",
        "\n",
        "time.sleep(5)\n",
        "\n",
        "\n",
        "ngrok.set_auth_token(\"2wJv717kfaZlPwHjKAAQMErod1t_5iYEcD4RaSpoLkHAwgz2i\")\n",
        "\n",
        "\n",
        "\n",
        "public_url = ngrok.connect(8501)\n",
        "print(public_url)\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": [],
      "authorship_tag": "ABX9TyNOHwa5tGSHpARPEINuaWII",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}