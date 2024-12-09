# coding: utf-8
# @email: enoche.chow@gmail.com

import torch
import pandas as pd
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os

def quick_inference(model, dataset, config_dict, save_model=True, mg=False):
    # Merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()

    # Log server and directory information
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # Load data
    dataset = RecDataset(config)
    logger.info(str(dataset))

    # Split data
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # Dataloaders
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    )

    # Manually specify hyperparams
    config['seed'] = 999
    config['reg_weight'] = 0.1
    config['learning_rate'] = 0.01

    logger.info('Running manually specified hyperparameter combination:')
    logger.info('Parameters: seed={}, reg_weight={}, learning_rate={}'.format(
        config['seed'], config['reg_weight'], config['learning_rate']))

    # Set seed
    init_seed(config['seed'])

    # Setup data loader random state
    train_data.pretrain_setup()

    # Model
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # Trainer
    trainer = get_trainer()(config, model, mg)

    # Training (optional)
    best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
        train_data, valid_data=valid_data, test_data=test_data, saved=save_model
    )

    # Results
    logger.info('Best validation result: {}'.format(dict2str(best_valid_result)))
    logger.info('Test result: {}'.format(dict2str(best_test_upon_valid)))
    logger.info('Finished running the manually specified combination.')

    # Save model if required
    if save_model:
        if not os.path.exists(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])
        model_save_path = os.path.join(config['checkpoint_dir'], f"{config['model']}_one_epoch.pth")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved at {model_save_path}")

    logger.info('Running inference...')

    # Build history_items_per_u from train_dataset
    user_field = config['USER_ID_FIELD']
    item_field = config['ITEM_ID_FIELD']
    history_items_per_u = {}
    for idx, row in train_dataset.df.iterrows():
        u = row[user_field]
        i = row[item_field]
        if u not in history_items_per_u:
            history_items_per_u[u] = set()
        history_items_per_u[u].add(i)

    # Load mappings
    user_mapping = pd.read_csv("/content/MMRec/data/u_id_mapping.csv", sep='\t')
    item_mapping = pd.read_csv("/content/MMRec/data/i_id_mapping.csv", sep='\t')

    # Dictionaries to revert back
    id2raw_user = {row["userID"]: row["user_id"] for _, row in user_mapping.iterrows()}
    id2raw_item = {row["itemID"]: row["asin"] for _, row in item_mapping.iterrows()}

    predictions = []
    model.eval()
    with torch.no_grad():
        num_batches = len(test_data)

        for batch_idx, batch in enumerate(test_data, start=1):
            logger.info(f"Processing batch {batch_idx}/{num_batches}...")

            users = batch[0]

            # Debugging: user history for first 5
            for user in users[:5]:
                user_id = user.item()
                user_history_internal = list(history_items_per_u.get(user_id, []))
                user_history_original = [id2raw_item[i] for i in user_history_internal]
                raw_user_id = id2raw_user[user_id]
                logger.info(f"User {raw_user_id} history (original IDs): {user_history_original}")

            # Extract true items if available
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].shape[0] == 2:
                mask_matrix = batch[1]
                true_items_for_all_users = []
                for local_user_idx, user_id in enumerate(users):
                    user_mask = (mask_matrix[0] == local_user_idx)
                    user_true_items = mask_matrix[1][user_mask]
                    true_items_for_all_users.append(user_true_items)
            else:
                true_items_for_all_users = [torch.tensor([], device=users.device) for _ in users]

            # For debugging: log true items for first 5 users
            for i in range(min(len(users), 5)):
                internal_true_items = true_items_for_all_users[i].cpu().tolist()
                raw_user_id = id2raw_user[users[i].item()]
                original_true_items = [id2raw_item[iid] for iid in internal_true_items]
                logger.info(f"User {raw_user_id} true items (original IDs): {original_true_items}")

            # Prediction
            interaction = (users, None)
            scores = model.full_sort_predict(interaction)
            logger.info(f"Score matrix shape: {scores.shape}")
            top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

            for idx, user in enumerate(users):
                internal_user_id = user.item()
                raw_user_id = id2raw_user[internal_user_id]

                # History
                user_history_internal = list(history_items_per_u.get(internal_user_id, []))
                user_history_original = [id2raw_item[i] for i in user_history_internal]

                # True items
                internal_true_items = true_items_for_all_users[idx].cpu().tolist()
                original_true_items = [id2raw_item[iid] for iid in internal_true_items]

                # Top-20 predictions
                internal_top20 = top_k_indices[idx].cpu().tolist()
                original_top20 = [id2raw_item[iid] for iid in internal_top20]
                top20_scores = top_k_scores[idx].cpu().tolist()

                # Store predictions (adjust as needed for CSV)
                # The user requests: user_id, user_history_original, original_top20
                predictions.append({
                    "user_id": raw_user_id,
                    "user_history_original": user_history_original,
                    "original_top20": original_top20
                })

            # Debugging first 5 users' predictions
            max_log_users = 5
            logger.info(f"Logging top-20 predictions (original IDs) for up to {max_log_users} users in batch {batch_idx}...")
            for user_idx, (user, top_scores_per_user, top_items_per_user) in enumerate(zip(users, top_k_scores, top_k_indices)):
                if user_idx >= max_log_users:
                    break
                internal_user_id = user.item()
                raw_user_id = id2raw_user[internal_user_id]

                user_history_internal = list(history_items_per_u.get(internal_user_id, []))
                user_history_original = [id2raw_item[i] for i in user_history_internal]

                internal_true_items = true_items_for_all_users[user_idx].cpu().tolist()
                original_true_items = [id2raw_item[iid] for iid in internal_true_items]

                internal_top20 = top_items_per_user.cpu().tolist()
                original_top20 = [id2raw_item[iid] for iid in internal_top20]

                logger.info(
                    f"User {raw_user_id}: "
                    f"History: {user_history_original} "
                    f"True Items: {original_true_items} "
                    f"Top-20: {original_top20} with Scores {top_scores_per_user.cpu().tolist()}"
                )

    logger.info("Inference completed.")

    # Convert predictions to a DataFrame and save as CSV
    df = pd.DataFrame(predictions, columns=["user_id", "user_history_original", "original_top20"])
    output_csv_path = "inference_results.csv"
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")




    # with torch.no_grad():
    #   num_batches = len(test_data)  # Total number of batches

    #   for batch_idx, batch in enumerate(test_data, start=1):
    #       logger.info(f"Processing batch {batch_idx}/{num_batches}...")

    #       # Extract users
    #       users = batch[0]  # `batch[0]` are the users in this batch
    #       logger.info(f"Users: {users[:5]}")  # Log the first 5 users for debugging

    #       # Extract true items using mask_matrix
    #       # batch[1] = mask_matrix of shape [2, N]
    #       # mask_matrix[0] = local user indices in this batch
    #       # mask_matrix[1] = corresponding true item IDs
    #       if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].shape[0] == 2:
    #           mask_matrix = batch[1]
    #           true_items_for_all_users = []
    #           for local_user_idx, user_id in enumerate(users):
    #               # Identify which entries in mask_matrix belong to this user
    #               user_mask = (mask_matrix[0] == local_user_idx)
    #               user_true_items = mask_matrix[1][user_mask]
    #               true_items_for_all_users.append(user_true_items)
    #       else:
    #           # If no true items available
    #           true_items_for_all_users = [torch.tensor([], device=users.device) for _ in users]

    #       # For debugging, log the first 5 users' true items
    #       for i in range(min(len(users), 5)):
    #           logger.info(f"User {users[i].item()} true items: {true_items_for_all_users[i].cpu().tolist()}")

    #       # Prepare interaction tuple for prediction (users only)
    #       interaction = (users, None)

    #       # Compute scores for all items for the given users
    #       scores = model.full_sort_predict(interaction)
    #       logger.info(f"Score matrix shape: {scores.shape}")

    #       # Extract top-20 items and scores for each user
    #       top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

    #       # Store predictions
    #       predictions.extend([
    #           {
    #               "user_id": user.item(),
    #               "true_items": true_items_for_all_users[idx].cpu().tolist(),
    #               "top_20_items": top_k_indices[idx].cpu().tolist(),
    #               "top_20_scores": top_k_scores[idx].cpu().tolist(),
    #           }
    #           for idx, user in enumerate(users)
    #       ])

    #       # Log the top-20 results for the first 5 users in the batch for debugging
    #       max_log_users = 5
    #       logger.info(f"Logging top-20 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #       for user_idx, (user, top_scores_per_user, top_items_per_user) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #           if user_idx >= max_log_users:
    #               break
    #           logger.info(
    #               f"User {user.item()}: "
    #               f"True Items {true_items_for_all_users[user_idx].cpu().tolist()} "
    #               f"Top-20 Items {top_items_per_user.cpu().tolist()} with Scores {top_scores_per_user.cpu().tolist()}"
    #           )

    # logger.info("Inference completed.")



    # with torch.no_grad():
    #     num_batches = len(test_data)  # Total number of batches

    #     for batch_idx, batch in enumerate(test_data, start=1):
    #         logger.info(f"Processing batch {batch_idx}/{num_batches}...")

    #         # Extract users and true items
    #         users = batch[0]  # Assuming `batch[0]` contains user indices
    #         logger.info(f"Users: {users[:5]}")  # Log the first 5 users for debugging

    #         if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].shape[0] >= 2:
    #             # Extract multiple true items for the users
    #             true_items = batch[1][1][users]  # Correctly map true items to users
    #         else:
    #             true_items = None  # Handle cases where true items are not available
    #         logger.info(f"Extracted true items: {true_items[:5] if true_items is not None else 'None'}")

    #         # Prepare interaction tuple for prediction
    #         interaction = (users, None)  # Pass users only for prediction

    #         # Compute scores for all items for the given users
    #         scores = model.full_sort_predict(interaction)
    #         logger.info(f"Score matrix shape: {scores.shape}")

    #         # Extract top-20 items and scores for each user in the batch
    #         top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

    #         # Store predictions along with user IDs, true items, and scores
    #         predictions.extend([
    #             {
    #                 "user_id": user.item(),
    #                 "true_items": true_items[idx].tolist() if true_items is not None else [],
    #                 "top_20_items": top_k_indices[idx].cpu().tolist(),
    #                 "top_20_scores": top_k_scores[idx].cpu().tolist(),
    #             }
    #             for idx, user in enumerate(users)
    #         ])

    #         # Log the top-20 results for the first 5 users in the batch for debugging
    #         max_log_users = 5
    #         logger.info(f"Logging top-20 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #         for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #             if user_idx >= max_log_users:
    #                 break
    #             logger.info(
    #                 f"User {user.item()}: "
    #                 f"True Items {true_items[user_idx].tolist() if true_items is not None else 'None'} "
    #                 f"Top-20 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #             )

    # logger.info("Inference completed.")





    # Inference: Run the saved model on the test dataset

    # ----------------------WORKING(ONLY SINGLE TRUE VALUE)------------------------------------------
    # logger.info('Running inference...')
    # predictions = []
    # model.eval()  # Ensure model is in evaluation mode
    # with torch.no_grad():
    #     num_batches = len(test_data)  # Total number of batches

    #     for batch_idx, batch in enumerate(test_data, start=1):
    #         logger.info(f"Processing batch {batch_idx}/{num_batches}...")

    #         # Extract users from the batch
    #         users = batch[0]  # Assuming `batch[0]` contains user indices
    #         logger.info(f"Users: {users[:5]}")  # Log the first 5 users for debugging

    #         # Extract true items for users (using corrected logic)
    #         if len(batch) > 1 and isinstance(batch[1], torch.Tensor) and batch[1].shape[0] >= 2:
    #             true_items = batch[1][1][users]  # Correct extraction of true items
    #         else:
    #             true_items = None  # Handle cases where true items are not available
    #         logger.info(f"Extracted true items: {true_items[:5] if true_items is not None else 'None'}")

    #         # Prepare interaction tuple for prediction
    #         interaction = (users, None)  # Pass users only for prediction

    #         # Compute scores for all items for the given users
    #         scores = model.full_sort_predict(interaction)
    #         logger.info(f"Score matrix shape: {scores.shape}")

    #         # Extract top-20 items and scores for each user in the batch
    #         top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

    #         # Store predictions along with user IDs, true items, and scores
    #         predictions.extend([
    #             {
    #                 "user_id": user.item(),
    #                 "true_items": true_items[idx].item() if true_items is not None else None,
    #                 "top_20_items": top_k_indices[idx].cpu().tolist(),
    #                 "top_20_scores": top_k_scores[idx].cpu().tolist(),
    #             }
    #             for idx, user in enumerate(users)
    #         ])

    #         # Log the top-20 results for the first 5 users in the batch for debugging
    #         max_log_users = 5
    #         logger.info(f"Logging top-20 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #         for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #             if user_idx >= max_log_users:
    #                 break
    #             logger.info(
    #                 f"User {user.item()}: "
    #                 f"True Item {true_items[user_idx].item() if true_items is not None else 'None'} "
    #                 f"Top-20 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #             )

    # logger.info("Inference completed.")
    #------------------------------------------------------------------------------------------------------------


   #-------------------------------------------------------------------------
     #---------------------DEBUGGING--------------------------------------------------
      # num_batches = len(test_data)
      # for batch_idx, batch in enumerate(test_data, start=1):
      #     logger.info(f"Processing batch {batch_idx}/{num_batches}...")
      #     users = batch[0]
      #     logger.info(f"Users: {users[:5]}")
      #     # Adjust true item extraction
      #     try:
      #         true_items = batch[1][1][users]  # Primary extraction method
      #     except IndexError as e:
      #         logger.warning(f"Fallback triggered due to: {e}")
      #         true_items = batch[1][users]

      #     logger.info(f"True items before alignment: {true_items[:5]}")
      #     true_items += model.n_users
      #     true_items = torch.clamp(true_items, min=0, max=model.result.size(0) - 1)
      #     logger.info(f"True items after alignment: {true_items[:5]}")

      #     # Proceed with inference
      #     interaction = (users, true_items)
      #     scores = model.full_sort_predict(interaction)
      #     logger.info(f"Score matrix shape: {scores.shape}")
      #     logger.info(f"Scores for user 0: {scores[0, :5]}")
    # #-------------------------------------------------------------------------
    # with torch.no_grad():
    #     num_batches = len(test_data)  # Total number of batches

    #     for batch_idx, batch in enumerate(test_data, start=1):
    #         try:
    #             logger.info(f"Processing batch {batch_idx}/{num_batches}...")

    #             # Extract users and their true positive items from the batch
    #             users = batch[0]  # Assuming `batch[0]` contains user indices
    #             true_items = batch[1]  # Assuming `batch[1]` contains true items for the current batch

    #             # Debug: Log raw batch data for inspection
    #             logger.info(f"Raw users in batch {batch_idx}: {users[:5]}")  # Log first 5 users
    #             logger.info(f"Raw true_items in batch {batch_idx}: {true_items[:5]}")  # Log first 5 true items
                
    #             # Debug: Log initial shapes and ranges
    #             logger.info(f"Initial users shape: {users.shape}")  # Should match batch size
    #             logger.info(f"Initial true items shape: {true_items.shape}")  # Should match batch size or full dataset shape

    #             # Adjust true items to match the current batch
    #             if true_items.dim() > 1:  # If true_items represents full dataset, filter by users
    #                 logger.info(f"True items before filtering by users: {true_items[:5]}")
    #                 true_items = true_items[0][users]  # Align true items with current batch users
    #                 logger.info(f"True items after filtering by users: {true_items[:5]}")

    #             # Map true items to the same embedding space as scores
    #             true_items += model.n_users  # Align with embedding space if necessary

    #             # Debug: Log true items after embedding alignment
    #             logger.info(f"True items after embedding alignment: {true_items[:5]}")

    #             # Prepare interaction tuple for prediction
    #             interaction = (users, true_items)

    #             # Compute scores for all items for the given users
    #             scores = model.full_sort_predict(interaction)  # Must be defined before clamping

    #             # Debug: Log range of true_items before clamping
    #             logger.info(f"True items range before clamping: {true_items.min().item()} to {true_items.max().item()}")

    #             # Validate scores dimensions
    #             assert scores.dim() == 2, "Scores must be a 2D tensor [batch_size, num_items]"
    #             max_valid_index = scores.size(1) - 1  # Max valid item index

    #             # Clamp true_items to valid range for scores
    #             true_items = torch.clamp(true_items, min=0, max=max_valid_index)

    #             # Debug: Log range of true_items after clamping
    #             logger.info(f"True items range after clamping: {true_items.min().item()} to {true_items.max().item()}")

    #             # Debug: Log unique values of true_items after clamping
    #             logger.info(f"Unique true_items in batch {batch_idx}: {true_items.unique()}")

    #             # Ensure correct shape for gather operation
    #             true_items_expanded = true_items.view(-1, 1)  # Reshape to (batch_size, 1)

    #             # Extract top-20 items and scores for each user in the batch
    #             top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

    #             # Gather scores for true positive items
    #             true_item_scores = scores.gather(1, true_items_expanded)

    #             # Debug: Log gathered true item scores
    #             logger.info(f"True item scores shape: {true_item_scores.shape}")  # Expected: [batch_size, 1]

    #             # Store predictions along with user IDs, true values, and true scores
    #             predictions.extend([
    #                 {
    #                     "user_id": user.item(),
    #                     "true_items": true_items[idx].cpu().tolist(),
    #                     "true_item_scores": true_item_scores[idx].cpu().tolist(),
    #                     "top_20_items": top_k_indices[idx].cpu().tolist(),
    #                     "top_20_scores": top_k_scores[idx].cpu().tolist(),
    #                 }
    #                 for idx, user in enumerate(users)
    #             ])

    #             # Log predictions for debugging
    #             max_log_users = 5
    #             logger.info(f"Logging predictions for up to {max_log_users} users in batch {batch_idx}...")
    #             for user_idx, (user, true_item, true_score, top_items, top_scores) in enumerate(
    #                 zip(users, true_items, true_item_scores.squeeze(1), top_k_indices, top_k_scores)
    #             ):
    #                 if user_idx >= max_log_users:
    #                     break
    #                 logger.info(
    #                     f"User {user.item()}: True Item {true_item.item()} with Score {true_score.item()}"
    #                 )
    #                 logger.info(
    #                     f"User {user.item()}: Top-20 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #                 )

    #         except Exception as e:
    #             logger.error(f"Error processing batch {batch_idx}: {e}")
    #             import traceback
    #             traceback.print_exc()  # Log detailed traceback for debugging

    # logger.info("Inference completed.")
    #---------------------------------------------------------------------------------------------
    
    # model.eval()  # Ensure model is in evaluation mode
    # with torch.no_grad():
    #   num_batches = len(test_data)  # Total number of batches

    #   for batch_idx, batch in enumerate(test_data, start=1):
    #       logger.info(f"Processing batch {batch_idx}/{num_batches}...")
    #       print("THIS IS THE BATCH",batch)
    #       # Extract users and user history from the batch
    #       users = batch[0]  # Assuming `batch[0]` contains user indices
    #       user_history = batch[1] if len(batch) > 1 else None  # Assuming `batch[1]` contains user history

    #       # Prepare interaction tuple for prediction
    #       interaction = (users, None)  # Modify if required

    #       # Compute scores for all items for the given users
    #       scores = model.full_sort_predict(interaction)

    #       # Extract top-20 items and scores for each user in the batch
    #       top_k_scores, top_k_indices = torch.topk(scores, k=20, dim=1)

    #       # Store predictions along with user IDs and history
    #       predictions.extend([
    #           {
    #               "user_id": user.item(),
    #               "user_history": user_history[:, idx].cpu().tolist() if user_history is not None else [],
    #               "top_20_items": top_k_indices[idx].cpu().tolist(),
    #               "top_20_scores": top_k_scores[idx].cpu().tolist(),
    #           }
    #           for idx, user in enumerate(users)
    #       ])

    #       # Log the top-20 results for the first 5 users in the batch for debugging
    #       max_log_users = 5
    #       logger.info(f"Logging top-20 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #       for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #           if user_idx >= max_log_users:
    #               break
    #           logger.info(
    #               f"User {user.item()}: "
    #               f"Top-20 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #           )
    #           if user_history is not None:
    #               logger.info(f"User {user.item()} History: {user_history[:, user_idx].cpu().tolist()}")

    # logger.info("Inference completed.")
      
    
    # with torch.no_grad():
    #   num_batches = len(test_data)  # Total number of batches
    #   predictions = []

    #   for batch_idx, batch in enumerate(test_data, start=1):
    #       logger.info(f"Processing batch {batch_idx}/{num_batches}...")

    #       # Extract user indices from the batch (adjust to your test_data format)
    #       users = batch[0]  # Assuming `batch[0]` contains user indices
          
    #       # Ensure the interaction input matches the expected structure for full_sort_predict
    #       interaction = (users, None)  # Modify `None` if items are also part of the batch

    #       # Compute scores for all items for the given users
    #       scores = model.full_sort_predict(interaction)

    #       # Extract top-5 items and scores for each user in the batch
    #       top_k_scores, top_k_indices = torch.topk(scores, k=5, dim=1)
    #       predictions.append((top_k_scores, top_k_indices))

    #       # Limit logging to first 5 users in the batch
    #       max_log_users = 5
    #       logger.info(f"Logging top-5 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #       for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #           if user_idx >= max_log_users:
    #               break
    #           logger.info(
    #               f"User {user.item()}: Top-5 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #           )

    #   logger.info("Inference completed.")

    #<----------------------------------------------------------------
    # num_batches = len(test_data)  # Total number of batches
    # with torch.no_grad():
    #   for batch_idx, batch in enumerate(test_data, start=1):
    #       logger.info(f"Processing batch {batch_idx}/{num_batches}...")
          
    #       # Extract user indices and compute scores for all items
    #       users = batch[0]  # Extract user indices from the batch
    #       scores = model.full_sort_predict(batch)  # Compute scores for all items
          
    #       # Extract top-5 items and scores for each user in the batch
    #       top_k_scores, top_k_indices = torch.topk(scores, k=5, dim=1)
    #       predictions.append((top_k_scores, top_k_indices))
          
    #       # Limit logging to first 5 users in the batch
    #       max_log_users = 5
    #       logger.info(f"Logging top-5 predictions for up to {max_log_users} users in batch {batch_idx}...")
    #       for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #           if user_idx >= max_log_users:
    #               break
    #           logger.info(
    #               f"User {user.item()}: Top-5 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}"
    #           )
    # logger.info("Inference completed.")
    #<----------------------------------------------------------------------

    # with torch.no_grad():
    #     for batch in test_data:
    #         print(batch)
    #         users = batch[0]
    #         scores = model.full_sort_predict(batch)
    #         predictions.append(scores)

    # # Log top 5 predictions
    # logger.info(f"Inference completed. Predictions (Top 5): {predictions[:5]}")
    # return predictions






    # with torch.no_grad():
    #   for batch_idx, batch in enumerate(test_data):
    #     logger.info(f"Processing batch {batch_idx + 1}...")
    #     print(batch)
    #     users = batch[0]  # Extract user indices from the batch
    #     scores = model.full_sort_predict(batch)  # Compute scores for all items
        
    #     # Extract top-5 items and scores for each user in the batch
    #     top_k_scores, top_k_indices = torch.topk(scores, k=5, dim=1)
    #     predictions.append((top_k_scores, top_k_indices))
        
    #     # Log the top-5 predictions for each user in the batch
    #     for user_idx, (user, top_scores, top_items) in enumerate(zip(users, top_k_scores, top_k_indices)):
    #         logger.info(f"User {user.item()}: Top-5 Items {top_items.cpu().tolist()} with Scores {top_scores.cpu().tolist()}")


    # logger.info("Inference completed.")
    # return predictions
    
    # <---------------------------------------
    # This is the OG that works, bring back to GRCN.py full predict and use this part
    # fix any part.
    # with torch.no_grad():
    #     for batch in test_data:
    #         print(batch)
    #         users = batch[0]
    #         scores = model.full_sort_predict(batch)
    #         predictions.append(scores)

    # # Log top 5 predictions
    # logger.info(f"Inference completed. Predictions (Top 5): {predictions[:5]}")
    # return predictions
    # <---------------------------------


    
    # hyper_ret = []
    # val_metric = config['valid_metric'].lower()
    # best_test_value = 0.0
    # idx = best_test_idx = 0

    # logger.info('\n\n=================================\n\n')

    # # hyper-parameters
    # hyper_ls = []
    # if "seed" not in config['hyper_parameters']:
    #     config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    # for i in config['hyper_parameters']:
    #     hyper_ls.append(config[i] or [None])
    # # combinations
    # combinators = list(product(*hyper_ls))
    # total_loops = len(combinators)
    # for hyper_tuple in combinators:
    #     # random seed reset
    #     for j, k in zip(config['hyper_parameters'], hyper_tuple):
    #         config[j] = k
    #     init_seed(config['seed'])

    #     logger.info('========={}/{}: Parameters:{}={}======='.format(
    #         idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

    #     # set random state of dataloader
    #     train_data.pretrain_setup()
    #     # model loading and initialization
    #     model = get_model(config['model'])(config, train_data).to(config['device'])
    #     logger.info(model)

    #     # trainer loading and initialization
    #     trainer = get_trainer()(config, model, mg)
    #     # debug
    #     # model training
    #     best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
    #     #########
    #     hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

    #     # save best test
    #     if best_test_upon_valid[val_metric] > best_test_value:
    #         best_test_value = best_test_upon_valid[val_metric]
    #         best_test_idx = idx
    #     idx += 1

    #     logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
    #     logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
    #     logger.info('████Current BEST████:\nParameters: {}={},\n'
    #                 'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
    #         hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # # log info
    # logger.info('\n============All Over=====================')
    # for (p, k, v) in hyper_ret:
    #     logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
    #                                                                               p, dict2str(k), dict2str(v)))

    # logger.info('\n\n█████████████ BEST ████████████████')
    # logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
    #                                                                hyper_ret[best_test_idx][0],
    #                                                                dict2str(hyper_ret[best_test_idx][1]),
    #                                                                dict2str(hyper_ret[best_test_idx][2])))

