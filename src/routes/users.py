from fastapi import APIRouter, HTTPException
from ..db.supabase import supabase_client
from ..types.user import ClaimRewardsRequest, DepositRequest
from ..utils.coinbase import call_contract_function, WalletType

router = APIRouter(
    prefix="/users",
    tags=["users"]
)
@router.get("/get-user-balance/{wallet_address}")
async def get_user_balance(wallet_address: str):
    try:
        user_balance = supabase_client.table("users").select("*").eq("wallet_address", wallet_address).execute()
        return {"status": "success", "data": user_balance.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get-user-sent-transactions/{wallet_address}")
async def get_user_sent_transactions(wallet_address: str):
    try:
        user_transactions = supabase_client.table("transactions").select("*").eq("from_wallet", wallet_address).execute()
        return {"status": "success", "data": user_transactions.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get-user-received-transactions/{wallet_address}")
async def get_user_received_transactions(wallet_address: str):
    try:
        user_transactions = supabase_client.table("transactions").select("*").eq("to_wallet", wallet_address).execute()
        return {"status": "success", "data": user_transactions.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/deposit")
async def deposit_tokens(wallet: WalletType, request: DepositRequest):
    try:
        deposit_result = call_contract_function(wallet, "mint", {"to": wallet.address_id, "amount": request.amount})
        if deposit_result.get('success') != True:
            raise HTTPException(status_code=500, detail="Deposit failed")
        user_balance = supabase_client.table("users").select("*").eq("wallet_address", wallet.address_id).execute()
        current_nrb = user_balance.data[0].get("nrb_value")
        amount = float(request.amount)
        supabase_client.table("users").update({"nrb_value": current_nrb + amount}).eq("wallet_address", wallet.address_id).execute()
        return {"status": "success", "message": "Tokens deposited successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/claim")
async def claim_tokens(wallet: WalletType, request: ClaimRewardsRequest):
    try:
        transaction_index = supabase_client.table("transactions")\
            .select("index, from_wallet")\
            .eq("id", request.transaction_id)\
            .execute()
        print(transaction_index.data[0].get("index"))
        print(transaction_index.data[0].get("from_wallet"))
        claim_result = call_contract_function(wallet, "withdrawLockedTokens", {"index": transaction_index.data[0].get("index"), "from": transaction_index.data[0].get("from_wallet")})
        if claim_result.get('success') != True:
            raise HTTPException(status_code=500, detail="Claim rewards failed")
        return {"status": "success", "message": "Tokens claimed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.get("/get-user-disputes/{wallet_address}")
async def get_user_disputes(wallet_address: str):
    try:
        # First get all transactions where user is sender
        user_transactions = supabase_client.table("transactions")\
            .select("id")\
            .eq("from_wallet", wallet_address)\
            .execute()

        # Get transaction IDs
        transaction_ids = [t["id"] for t in user_transactions.data]
        
        # Get all disputes for these transactions
        disputes = supabase_client.table("disputes")\
            .select("*")\
            .in_("transactionId", transaction_ids)\
            .execute()

        return {"status": "success", "data": disputes.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
