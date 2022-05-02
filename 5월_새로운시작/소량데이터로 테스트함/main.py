import seed
import data_processing
import model_train
import torch
import Decoder
import save_excel

seed.set_seed(42)
file_name = "t5Large, trainData_1000, entailment, nucleus_sampling, length_200"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

entailment, neutral, contradiction = data_processing.data_load()

model = model_train.model_train(
    device=device,
    dataset= entailment[:1000],
    epochs=5,
    path=file_name,
    set_max_length= 200,
    model_name='t5-large'
)

outputs = Decoder.generation_sentence(model=model,
                                                  dataset=entailment[1000:1200],
                                                  device=device,
                                                  decoder_argorithm="nucleus_sampling",  #beam_search, nucleus_sampling
                                                  model_name = 't5-large',
                                                  setting_length= 200
                                                  )

save_excel.save_csv(outputs, file_name)