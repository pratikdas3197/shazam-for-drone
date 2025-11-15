# shazam-for-drone
Scalable Drone Audio Identification like Shazam

## ✨ Features

* **Feature 1:** Create a database of drone audio embeddings
* **Feature 2:** Match drone audio to the database
* **Feature 3:** Plot the emebddings of the audio files

---

## 🛠️ Installation and Setup

### 1. Clone the Repository

First, clone the project files to your local machine using git:

```bash
git clone https://github.com/pratikdas3197/shazam-for-drone.git
cd shazam-for-drone
```

### 2. Create and Activate a Virtual Environment (venv)

It's highly recommended to use a **virtual environment** to isolate your project's dependencies from your system's global Python packages. This prevents conflicts and keeps your project reproducible.

#### **Creating the `venv`**

Run the following command in your terminal from the project's root directory. This creates a new folder named `.venv` (or `venv`) inside your project that contains a separate Python installation.

```bash
python3 -m venv .venv
```

#### **Activating the `venv`**

After creation, you must **activate** the virtual environment. This tells your terminal to use the Python and `pip` (package manager) executables inside the `.venv` folder instead of the system-wide ones.

* **On macOS/Linux:**

    ```bash
    source .venv/bin/activate
    ```

* **On Windows (Command Prompt):**

    ```bash
    .venv\Scripts\activate.bat
    ```

* **On Windows (PowerShell):**

    ```bash
    .venv\Scripts\Activate.ps1
    ```

> 💡 **Tip:** Once activated, you'll usually see the name of your environment (e.g., `(.venv)`) at the beginning of your terminal prompt. To **deactivate** it later, simply type `deactivate` and press Enter.

### 3. Install Dependencies

With the virtual environment activated, install all required packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Embedding Script

To run the main application file:

```bash
python save_drones.py
```

### Running the Matching Script

```bash
python find_drone.py
```

---

## 📝 License

This project is licensed under the **[MIT License](LICENSE)** - see the `LICENSE` file for details.
EOF