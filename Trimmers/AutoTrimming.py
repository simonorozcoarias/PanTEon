import os, re
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from Bio import SeqIO
import subprocess
import zipfile
import multiprocessing
import shutil
from pickle import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.base import TransformerMixin
import math


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

TE_size=15000

# Downloads a specific genome for a given species
def download_genome(species_genome, zip_file, env_name="autotrim_env"):
	try:
		# Attempts to download using '--reference' option
		print(f"Trying downloading {species_genome} with --reference...")

		# Running the command as in the terminal
		subprocess.run(["conda", "run", "-n", env_name, "datasets", "download",
						"genome", "taxon", species_genome, "--reference", "--filename", zip_file],
					   check=True, capture_output=True, text=True
					   )

		print(f"Download with --reference completed for {species_genome}.")

	except subprocess.CalledProcessError as e:
		try:

			# Attempts to download without '--reference' option
			print(f"Error with --reference. Retrying without --reference for {species_genome}...")

			# Running the command as in the terminal
			subprocess.run(["conda", "run", "-n", env_name, "datasets", "download",
							"genome", "taxon", species_genome, "--filename", zip_file],
						   check=True, capture_output=True, text=True
						   )

			print(f"Download without --reference completed for {species_genome} {e}.")

		except subprocess.CalledProcessError:
			# Checks if this also fails
			print(f"Error: Download not possible for {species_genome}.")
			print(f"STDERR:\n{e.stderr}")
			return False

	return True


# Unzips a genome .zip file in the genome directory
def unzip_genome(zip_file, genome_dir):
	if not os.path.exists(zip_file):
		print(f"Error: File {zip_file} was not found")
		return False

	try:
		# Create genome folder it if doesnt exist
		os.makedirs(genome_dir, exist_ok=True)

		# Unzip the file
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall(genome_dir)

		print(f"File {zip_file} was unzipped in {genome_dir}")
		return True

	except zipfile.BadZipFile:
		print(f"Error: {zip_file} is not a valid .zip file or it is corrupted.")
		return False

	except Exception as e:
		print(f"Unexpected error while unzipping {zip_file}: {e}")
		return False


# Saves the path of the .fna genome file
def find_fna_file(genome_dir, species_name):
	fna_file = None

	# Checks if genome directory exists
	if not os.path.exists(genome_dir):
		print(f"Error: genome directory {genome_dir} does not exist")
		return None

	# Search recursively for the first .fna file
	for root, dirs, files in os.walk(genome_dir):
		for file in files:
			if file.endswith(".fna"):
				fna_file = os.path.join(root, file)
				break
		if fna_file:
			break

	# If found, returns the path to the fna file
	if fna_file:
		print(f"Genome of {species_name} downloaded and unzipped.")
		print(f"File .fna found in: {fna_file}")
		return fna_file

	else:
		print(f"Error: File .fna for {species_name} couldnt be found.")
		return None


# Gets fasta header and runs TE-Aid
def run_extract_and_teaid(header, te_fasta, fna_file, output_dir, TEAid_dir="TEAid", env_name="autotrim_env"):
	# Directory of TE-Aid program and R scripts (by default, "TEAid")
	TEAid_dir = os.path.abspath(os.path.join(TEAid_dir, "TE-Aid"))

	# Checks if TE-Aid program exists
	if not os.path.isfile(TEAid_dir):
		print("TE-Aid was not found.")
		return False

	# Checks if FNA file exists
	if not os.path.isfile(fna_file):
		print(f"FNA file not found: {fna_file}")
		return False

	# Checks if FNA is already indexed
	if not os.path.exists(f"{fna_file}.nhr"):
		output = subprocess.run(
				['makeblastdb', '-in', fna_file, '-dbtype', 'nucl'], stdout=subprocess.PIPE, text=True)

	# Verify if TE fasta is not empty and exists
	if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
		print(f"Sequence not found for {header}")
		return False

	print("Sequence was extracted succesfully.")

	# Execute TE-Aid
	print(f"Running TE-Aid with TE fasta:{te_fasta} and FNA file:{fna_file}...")
	try:
		result = subprocess.run(
			[
				"conda", "run", "-n", env_name, TEAid_dir,
				"-q", os.path.abspath(te_fasta),
				"-g", os.path.abspath(fna_file),
				"-o", os.path.abspath(output_dir)
			],
			check=True,
			capture_output=True,
			text=True
		)

		print("stdout:", result.stdout)
		print("stderr:", result.stderr)

		print(f"TE-Aid completed successfully. Results saved in: {output_dir}")

	except subprocess.CalledProcessError as e:
		print(f"Unexpected error while running TE-Aid: {e}")
		print("stdout:", e.stdout)
		print("stderr:", e.stderr)
		return False

	return True


# Create species dictionary with indexes
def create_species_dict_from_fasta(input_fasta):
	species_dict = {}

	for idx, record in enumerate(SeqIO.parse(input_fasta, "fasta")):

		# Extract species after '@'
		match = re.search(r'@\s*([A-Z][a-z]+(?:\s+[a-z]+)+)', record.description)

		if match:
			species_name = match.group(1)
			species_dict.setdefault(species_name, []).append(idx)
		else:
			print(f"Warning: No species found for '{record.id}'")

	return species_dict


# Process a species from the dictionary (this includes downloading genome, running TE-Aid and getting the images)
def process_species(species, sequences, positions, headers, TEAid_dir, output_dir, genomes_dir):
	print(f"Processing species: {species}")
	species_safe = species.replace("_", " ")
	genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{species.replace(' ', '_')}_genome"))
	zip_file = os.path.abspath(os.path.join(genomes_dir, f"{species.replace(' ', '_')}.zip"))

	# Loops through the positions for a species
	for position in positions:
		try:
			header = headers[position]
			match_case = re.match(r'^>?([^#\s]+)', header)
			case_name = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

			# Create output directory for the case
			case_dir = os.path.join(output_dir, case_name)
			os.makedirs(case_dir, exist_ok=True)

			# New name for the PDF file (ends in .pdf)
			new_pdf = os.path.join(output_dir, f"{case_name}.pdf")

			# TE FASTA file
			te_fasta = os.path.join(case_dir, f"{case_name}.fasta")

			print(f"genome_dir: {genome_dir}")
			print(f"case_dir: {case_dir}")
			print(f"case_name: {case_name}")
			print(f"header: {header}")
			print(f"position: {position}")

			# Extracts sequence with matching header and saves it in TE fasta
			with open(te_fasta, "w") as f:
				f.write(f"{header}\n{sequences[position].seq}\n")

			# Check if pdf exists
			print(f"Checking the PDF file existence: {new_pdf}")
			if os.path.exists(new_pdf):
				print(f"PDF already exists: {new_pdf}")

				# Removes case directory
				shutil.rmtree(case_dir)
				continue

			if not os.path.exists(zip_file):
				download_genome(species_safe, zip_file)

			if not os.path.exists(genome_dir):
				unzip_genome(zip_file, genome_dir)

			fna_file = find_fna_file(genome_dir, species)
			if not fna_file:
				print(f"FNA file not found for species {species}")
				continue

			# Execute TE-Aid
			run_extract_and_teaid(header, te_fasta, fna_file, case_dir, TEAid_dir, env_name="autotrim_env")

			original_pdf = os.path.join(case_dir, f"{case_name}.fasta.c2g.pdf")

			# Checks if pdf file with original name exists and renames it
			if os.path.exists(original_pdf):
				os.rename(original_pdf, new_pdf)
				shutil.rmtree(case_dir)
				print(f"PDF renamed as: {new_pdf}")
			else:
				print(f"PDF was not found: {original_pdf}")

		except Exception as e:
			print(f"ERROR processing species:{species}. {e}")

	# Removing genome
	if genome_dir and os.path.exists(genome_dir) and new_pdf and os.path.exists(new_pdf):
		shutil.rmtree(genome_dir)
		if zip_file and os.path.exists(zip_file):
			os.remove(zip_file)
		print(f"Genome {species} deleted to save disk space.")


# Apply multiprocessing to process several species simultaneously
def generation_multiprocessing(input_fasta, TEAid_dir, n_processes, output_dir, genomes_dir):
	# Get headers of sequences
	sequences = list(SeqIO.parse(input_fasta, "fasta"))
	headers = [f">{sequence.description}" for sequence in sequences]

	total = len(headers)
	print(f"[INFO] {total} sequences were found in {input_fasta}")

	species_dict = create_species_dict_from_fasta(input_fasta)
	print(f"A total of {len(species_dict)} unique species were detected.")

	# Create processes
	processes = []
	for species, positions in species_dict.items():
		p = multiprocessing.Process(
			target=process_species,
			args=(species, sequences, positions, headers, TEAid_dir, output_dir, genomes_dir)
		)
		processes.append(p)

	# Run processes in batches of n_processes
	for i in range(0, len(processes), n_processes):
		batch = processes[i:i + n_processes]

		print(f"Initiating batch {i // n_processes + 1} of {(len(processes) + n_processes - 1) // n_processes} "
			  f"({len(batch)} processes in parallel)...")

		for p in batch:
			p.start()

		for p in batch:
			p.join()

		print(f"Batch {i // n_processes + 1} completed.\n")

	print("Processing was completed.")

	if os.path.exists(genomes_dir):
		shutil.rmtree(genomes_dir)

	if os.path.exists("db"):
		shutil.rmtree("db")


def generate_te_images(input_fasta, teaid_dir):
	import fitz
	# List with FASTA headers
	TEs = list(SeqIO.parse(input_fasta, "fasta"))
	te_image_info = []

	for TE in TEs:
		TE_name = TE.id.split("#")[0]
		species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
		species_name = species_match.group(0) if species_match else None

		pdf_path = os.path.join(teaid_dir, TE_name + '.pdf')
		if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) <= 4 * 1024:
			print(f"PDF not found for {TE_name}, continuing.")
			continue

		print(f"Generating image for TE_name: {TE_name}")
		try:
			doc = fitz.open(pdf_path)
			image_path = os.path.join(teaid_dir, TE_name + ".fa.c2g.jpeg")

			# Save first page as JPEG (adjust if you want multiple pages)
			for page_index in range(len(doc)):
				page = doc.load_page(page_index)
				pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
				pix.save(image_path, 'JPEG')

			te_image_info.append({
				"TE": TE,
				"TE_name": TE_name,
				"species_name": species_name,
				"image_path": image_path
			})
		except Exception as ex:
			print(f"Something went wrong with {TE_name}: {ex}")

	return te_image_info


# This class flattens N-dimensional data to 2D before applying scikit-learn's StandardScaler,
# and then reshapes the data back to its original dimensions after transformation.
class NDStandardScaler(TransformerMixin):
	def __init__(self, **kwargs):
		self._scaler = StandardScaler(copy=True, **kwargs)
		self._orig_shape = None

	def fit(self, X, **kwargs):
		X = np.array(X)
		# Save the original shape to reshape the flattened X later
		# back to its original shape
		if len(X.shape) > 1:
			self._orig_shape = X.shape[1:]
		X = self._flatten(X)
		self._scaler.fit(X, **kwargs)
		return self

	def transform(self, X, **kwargs):
		X = np.array(X)
		X = self._flatten(X)
		X = self._scaler.transform(X, **kwargs)
		X = self._reshape(X)
		return X

	def _flatten(self, X):
		# Reshape X to <= 2 dimensions
		if len(X.shape) > 2:
			n_dims = np.prod(self._orig_shape)
			X = X.reshape(-1, n_dims)
		return X

	def _reshape(self, X):
		# Reshape X back to it's original shape
		if len(X.shape) >= 2:
			X = X.reshape(-1, *self._orig_shape)
		return X

	def save_model(self, model_name):
		dump(self._scaler, open(model_name + '.bin', 'wb'))

	def load_model(self, model_path, X):
		self._scaler = load(open(model_path, 'rb'))
		self._orig_shape = X.shape[1:]


def r2_score(y_true, y_pred):
	# Sum of squared residuals
	sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

	# Mean of the true values
	mean_y_true = tf.reduce_mean(y_true)

	# Total sum of squares
	sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

	# Avoid division by zero
	epsilon = tf.keras.backend.epsilon()

	sum_squares = tf.maximum(sum_squares, epsilon)

	# Coefficient of determination (R2)
	R2 = 1 - sum_squares_residuals / sum_squares
	return R2


# load the pre-generated dataset
def load_data(fasta_path, work_dir, inference=False):
	if os.path.exists(f"{work_dir}/features_data.npy") and os.path.exists(f"{work_dir}/case_labels.npy"):
		# Load arrays using memory mapping
		features_data = np.load(f"{work_dir}/features_data.npy",mmap_mode="r")
		if not inference:
			labels_data = np.load(f"{work_dir}/labels_data.npy",mmap_mode="r")
		case_names = np.load(f"{work_dir}/case_labels.npy",mmap_mode="r")

		# Map each case name to its position in the NumPy arrays
		case_to_index = {
			case_name: index
			for index, case_name in enumerate(case_names)
		}

		selected_indices = []
		missing_cases = []

		for TE in SeqIO.parse(fasta_path, "fasta"):
			match_case = re.match(r"^>?([^#\s]+)", TE.id)
			case_name = (match_case.group(1) if match_case else re.sub(r"\W+", "_", TE.id.strip()))

			if case_name in case_to_index:
				selected_indices.append(case_to_index[case_name])
			else:
				missing_cases.append(case_name)

		if selected_indices:
			selected_indices = np.asarray(selected_indices, dtype=np.int64)

			# Extract the rows corresponding to the sequences in the input FASTA
			X = np.asarray(features_data[selected_indices])
			if not inference:
				Y = np.asarray(labels_data[selected_indices])
			else:
				Y = case_names[selected_indices].tolist()
		else:
			# Preserve the expected dimensionality when no sequence matches
			X = np.empty((0, *features_data.shape[1:]), dtype=features_data.dtype)
			if not inference:
				Y = np.empty((0, *labels_data.shape[1:]), dtype=labels_data.dtype)
			else:
				Y = np.empty((0, *case_names.shape[1:]), dtype=case_names.dtype)

		if missing_cases:
			print(
				f"Warning: {len(missing_cases)} sequences from the input FASTA "
				f"were not found in case_labels.npy."
			)

		return X, Y

	else:
		print(
			f"[ERROR] File {work_dir}/features_data.npy, {work_dir}/labels_data.npy, or "
			f"{work_dir}/case_labels.npy was not found. Please check whether the dataset has already been generated "
			"using: python3 Trimmers/AutoTrimming2.py --mode dataset ..."
		)
		return None, None


# Return a model for image plots
def get_model(input_size=(256, 256, 1), num_classes=128):
	cnn_div = cnn_branch(input_size, 1)
	cnn_cov = cnn_branch(input_size, 2)
	cnn_dot = cnn_branch(input_size, 3)
	cnn_str = cnn_branch(input_size, 4)
	model = auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str)

	return model


# Train a model with specified callbacks
def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
	# Reduce learning rate if val_loss doesn't improve
	lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
		monitor='val_loss',
		factor=0.01,
		patience=10,
		verbose=1
	)

	# Early stopping
	early_stopping = EarlyStopping(
		monitor='val_loss',
		patience=20,
		restore_best_weights=True,
		verbose=1
	)

	X_train = X_train.astype("float32")
	Y_train = Y_train.astype("float32")
	X_dev = X_dev.astype("float32")
	Y_dev = Y_dev.astype("float32")

	batch_size = min(batch_size, X_train.shape[0], X_dev.shape[0])
	train_steps = math.ceil(X_train.shape[0] / batch_size)
	val_steps = math.ceil(X_dev.shape[0] / batch_size)

	train_ds = (tf.data.Dataset.from_tensor_slices(((X_train[:, :, :, 0], X_train[:, :, :, 1], X_train[:, :, :, 2], X_train[:, :, :, 3]), Y_train))
				.shuffle(min(len(X_train), 10000), reshuffle_each_iteration=True)
				.batch(batch_size, drop_remainder=True)
				.repeat()
				.prefetch(tf.data.AUTOTUNE))

	val_ds = (tf.data.Dataset.from_tensor_slices(((X_dev[:, :, :, 0], X_dev[:, :, :, 1], X_dev[:, :, :, 2], X_dev[:, :, :, 3]), Y_dev))
			  .batch(batch_size, drop_remainder=True)
			  .repeat()
			  .prefetch(tf.data.AUTOTUNE))

	history = model.fit(
		train_ds,
		epochs=num_epochs,
		steps_per_epoch=train_steps,
		validation_data=val_ds,
		validation_steps=val_steps,
		callbacks=[lr_scheduler, early_stopping],
		verbose=1
	)

	del train_ds
	del val_ds

	return history


# Builds dataset arrays (features, labels, case/species names) from TE images
def build_dataset_from_images(input_fasta, output_dir, inference=False):
	import cv2
	# Save TE ids from input FASTA file
	TEs = {TE.id: TE for TE in SeqIO.parse(input_fasta, "fasta")}

	# Make a list with names of the image files
	image_files = [f for f in os.listdir(output_dir) if f.endswith(".jpeg")]

	# Select good images (size > 0)
	good_images = [f for f in image_files if os.path.getsize(os.path.join(output_dir, f)) > 0]

	print(f"Found {len(good_images)} valid images out of {len(image_files)} total.")

	# Initialize Numpy matrices and lists to store the data
	feature_data = np.zeros((len(good_images), 256, 256, 4), dtype=np.uint8)
	labels = np.zeros((len(good_images), 2), dtype=np.float32)
	case_names = []
	species_names = []

	n = 0

	for image_file in good_images:

		image_path = os.path.join(output_dir, image_file)

		# Extract TE_name from image file's name (remove extension)
		TE_name_match = re.match(r"(.+)\.fa\.c2g\.jpeg$", image_file)

		# Check if file exists
		if TE_name_match is None:
			print(f"Skipping unrecognized file: {image_file}")
			continue

		TE_name = TE_name_match.group(1)

		# Search for TE_name if TE list
		TE = next(
			(v for k, v in TEs.items() if k.startswith(TE_name)),
			None
		)

		if TE is None:
			print(f"Warning: TE {TE_name} not found in FASTA")
			continue

		# Get species from TE id
		species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
		species_name = species_match.group(0) if species_match else None

		# Transform image into grayscale
		te_aid_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		if te_aid_image is None:
			print(f"ERROR: Could not open image {TE_name}")
			continue

		# Split and resize plots, and save into different channels of feature data array
		feature_data[n, :, :, 0] = cv2.resize(te_aid_image[150:1030, 150:1130], (256, 256))
		feature_data[n, :, :, 1] = cv2.resize(te_aid_image[150:1030, 1340:2320], (256, 256))
		feature_data[n, :, :, 2] = cv2.resize(te_aid_image[1340:2220, 150:1130], (256, 256))
		feature_data[n, :, :, 3] = cv2.resize(te_aid_image[1340:2220, 1340:2320], (256, 256))

		# Starting and end position labels
		if not inference:
			start_pos = float(TE.id.split(" ")[0].split("#")[0].split("_")[-2])
			end_pos = float(TE.id.split(" ")[0].split("#")[0].split("_")[-1])
			labels[n, 0] = start_pos
			labels[n, 1] = min(end_pos, 1)

		# Append case name and species to lists
		case_names.append(TE_name)
		species_names.append(species_name)

		print(f"Processed {TE_name} -> n: {n}")
		n += 1

	# Save arrays
	np.save(os.path.join(output_dir, "features_data.npy"), feature_data[:n])
	np.save(os.path.join(output_dir, "case_labels.npy"), np.array(case_names))
	np.save(os.path.join(output_dir, "species_labels.npy"), np.array(species_names))
	if not inference:
		np.save(os.path.join(output_dir, "labels_data.npy"), labels[:n])
	else:
		labels = case_names


	print(f"Dataset created with {n} TEs.")
	return feature_data, labels


def cnn_branch(input_size, i):
	# Inputs
	inputs = tf.keras.Input(shape=input_size, name="input_" + str(i))

	# layer 1
	layers = tf.keras.layers.Conv2D(16, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
									kernel_initializer='he_uniform',
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
									padding="same", name="conv_" + str(i) + "_1")(inputs)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_1")(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.8, epsilon=0.001, scale=False,
												name="BN_" + str(i) + "_1")(layers)
	layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_1")(layers)

	# layer 2
	layers = tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
									kernel_initializer='he_uniform',
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
									padding="same", name="conv_" + str(i) + "_2")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_2")(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.4, epsilon=0.001, scale=False,
												name="BN_" + str(i) + "_2")(layers)
	layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_2")(layers)

	# layer 3
	layers = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
									kernel_initializer='he_uniform',
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
									padding="same", name="conv_" + str(i) + "_3")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_3")(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.2, epsilon=0.001, scale=False,
												name="BN_" + str(i) + "_3")(layers)
	layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_3")(layers)

	# layer 4
	layers = tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
									kernel_initializer='he_uniform',
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
									padding="same", name="conv_" + str(i) + "_4")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_4")(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.2, epsilon=0.001, scale=False,
												name="BN_" + str(i) + "_4")(layers)
	layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_4")(layers)

	# layer 5
	layers = tf.keras.layers.Flatten(name="flatten_" + str(i) + "_1")(layers)

	# layer end
	predictions = tf.keras.layers.Dense(128, activation="sigmoid", name="outputF_" + str(i))(layers)
	# model generation
	model = tf.keras.Model(inputs=inputs, outputs=predictions)
	return model


def auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str):
	combinedInput = tf.keras.layers.concatenate([cnn_div.output, cnn_cov.output, cnn_dot.output, cnn_str.output])

	# layer 1
	layers = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
								   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
								   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
								   name="dense_5_1")(combinedInput)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_1")(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
												)(layers)
	# layer 2
	layers = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
								   kernel_regularizer=tf.keras.regularizers.l1(0.001),
								   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
								   name="dense_5_2")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_2")(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
												)(layers)
	# layer 3
	layers = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
								   kernel_regularizer=tf.keras.regularizers.l1(0.001),
								   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
								   name="dense_5_3")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_3")(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
												)(layers)
	# layer 4
	layers = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
								   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
								   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
								   name="dense_5_4")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_4")(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
												)(layers)
	# layer 5
	layers = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
								   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
								   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
								   name="dense_5_5")(layers)
	layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_5")(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
												)(layers)

	# layer 4
	predictions = tf.keras.layers.Dense(2, activation="sigmoid", name="output_5")(layers)

	model = tf.keras.Model(inputs=[cnn_div.input, cnn_cov.input, cnn_dot.input, cnn_str.input], outputs=predictions)
	# optimizer
	opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

	# Compile model
	model.compile(loss="mse", optimizer=opt, metrics=[r2_score])
	return model


# ====================
# MAIN
# ====================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["dataset"], required=True,
						help="Execution Mode.")
	parser.add_argument("--input_fasta", help="Path to the TE library fasta file")
	parser.add_argument("--inference", default=False, help="Generating dataset for Inference mode? Default: False")
	parser.add_argument("--TEAid_dir", help="Path to the program TE+Aid")
	parser.add_argument("--processes", type=int, default=20, help="Number of threads to be used in the execution")
	parser.add_argument("--output_dir", default="te_aid", help="Path to the output directory")

	args = parser.parse_args()

	if args.mode == "dataset":
		# Transform the TE sequences into images (by extracting the species, downloading the genome,
		# running TE+Aid, and converting the PDF files to images in jepg format).
		# The output are the following numpy files (.npy) saved in the args.output_dir directory:
		# features_data.npy, labels_data.npy, case_labels.npy, and species_labels.npy.
		os.makedirs(args.output_dir, exist_ok=True)
		genomes_dir = f"{args.output_dir}/genomes"
		os.makedirs(genomes_dir, exist_ok=True)

		generation_multiprocessing(args.input_fasta, args.TEAid_dir, args.processes, args.output_dir, genomes_dir)

		# Generate images from TEAid .pdfs
		generate_te_images(args.input_fasta, args.output_dir)
		print("Images generated successfully.")

		build_dataset_from_images(args.input_fasta, args.output_dir, args.inference)

