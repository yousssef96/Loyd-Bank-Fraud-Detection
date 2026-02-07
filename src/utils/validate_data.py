import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator
from typing import Tuple, List

def validate_Lloyd_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Lloyd Bank Customer loan dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("Starting data validation with Great Expectations...")

    context = gx.get_context(mode="ephemeral")

    # 3. Wrap DataFrame in Validator via a Batch
    batch = Batch(data=df)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[batch],
        data_context=context,
    )

    # === Schema Validation - Essential Columns ===
    print("Validating schema and required columns...")

    # Customer identifier must exist
    validator.expect_column_to_exist("id")
    validator.expect_column_values_to_not_be_null("id")

    # Demographics feature
    validator.expect_column_to_exist("addr_state")
    validator.expect_column_to_exist("home_ownership")

    # Employment features
    validator.expect_column_to_exist("emp_title")
    validator.expect_column_to_exist("emp_length")

    # Financial Basics
    validator.expect_column_to_exist("annual_inc")
    validator.expect_column_to_exist("installment")
    validator.expect_column_to_exist("loan_amnt")
    validator.expect_column_to_exist("purpose")
    validator.expect_column_to_exist("term")
    validator.expect_column_to_exist("int_rate")

    # Account Balances and Activity
    validator.expect_column_to_exist("avg_cur_bal")
    validator.expect_column_to_exist("inq_last_12m")
    validator.expect_column_to_exist("max_bal_bc")
    validator.expect_column_to_exist("total_bal_ex_mort")

    # Account Age and Delinquency
    validator.expect_column_to_exist("mo_sin_old_il_acct")
    validator.expect_column_to_exist("mo_sin_old_rev_tl_op")
    validator.expect_column_to_exist("mo_sin_rcnt_rev_tl_op")
    validator.expect_column_to_exist("mo_sin_rcnt_tl")
    validator.expect_column_to_exist("mths_since_last_delinq")

    # Credit Line Metrics
    validator.expect_column_to_exist("mort_acc")
    validator.expect_column_to_exist("num_bc_tl")
    validator.expect_column_to_exist("num_il_tl")
    validator.expect_column_to_exist("num_op_rev_tl")
    validator.expect_column_to_exist("num_tl_90g_dpd_24m")
    validator.expect_column_to_exist("num_tl_op_past_12m")
    validator.expect_column_to_exist("open_acc")
    validator.expect_column_to_exist("total_acc")

    # Risk Indicators
    validator.expect_column_to_exist("percent_bc_gt_75")
    validator.expect_column_to_exist("pub_rec_bankruptcies")

    # === BUSINESS LOGIC VALIDATION ===
    print("Validating business logic constraints...")

    # Loan term must be on expected values 
    validator.expect_column_values_to_be_in_set("term", [" 36 months", " 60 months"])

    # Employment Length must be valid
    validator.expect_column_values_to_be_in_set(
        "emp_length",
        ['< 1 year', '1 year', '2 years', '3 years', '4 years',
         '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
         )
    
    # Home ownership must be valid
    validator.expect_column_values_to_be_in_set(
        "home_ownership",
        ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER']
        )
    
    # Loan purpose expected values
    validator.expect_column_values_to_be_in_set(
        "purpose",
        ['debt_consolidation', 'credit_card', 'other', 'home_improvement',
       'major_purchase', 'wedding', 'car', 'medical', 'small_business',
       'moving', 'house', 'vacation', 'renewable_energy', 'educational']
        )
    
    # Ensure Employment tile is actually text and not numeric
    validator.expect_column_values_to_be_of_type("emp_title", "object")


    # === GEOGRAPHIC VALIDATION ===


    # Based on the unique count of the 51 US states
    us_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ]
    
    validator.expect_column_values_to_be_in_set(
        "addr_state", 
        us_states
    )


    # === NUMERIC RANGE VALIDATION ===
    print("Validating numeric ranges and business constraints...")

    # Loan Amount min 1,000 
    validator.expect_column_values_to_be_between(
        "loan_amnt", min_value=1000, max_value=40000
    )
    # Annual Income must be positive
    validator.expect_column_values_to_be_between(
        "annual_inc", min_value=0, max_value=3000000
    )

    # Interest Rate 5.31% to 30.99% 
    validator.expect_column_values_to_be_between(
        "int_rate", min_value=0.05, max_value=0.32
    )

    # Monthly Installment 30.65 to 1503.89 in the data
    validator.expect_column_values_to_be_between(
        "installment", min_value=20, max_value=1600
    )

    # Credit History Ages (Months)
    validator.expect_column_values_to_be_between(
        "mo_sin_old_rev_tl_op", min_value=-1, max_value=800
    )
    validator.expect_column_values_to_be_between(
        "mo_sin_old_il_acct", min_value=-1, max_value=600
    )

    # Inquiry and Delinquency Counts
    validator.expect_column_values_to_be_between("inq_last_12m", min_value=-1, max_value=30)
    validator.expect_column_values_to_be_between("pub_rec_bankruptcies", min_value=-1, max_value=10)

    # === DATA CONSISTENCY CHECKS ===
    print("Validating loan business logic...")

    # Loan Status must be one of the expected classes
    validator.expect_column_values_to_be_in_set(
        "loan_status",
        ["Fully Paid", "Charged Off"]
    )

    

    # === RUN AND PROCESS ===
    print("Executing validation...")
    results = validator.validate()

    
    failed_expectations = [
        r.expectation_config.type 
        for r in results.results if not r.success
        ]

    if results["success"]:
        print(f"PASSED: {len(results['results'])} checks clean.")
    else:
        print(f"FAILED: Found {len(failed_expectations)} issues: {failed_expectations}")

    return results["success"], failed_expectations



