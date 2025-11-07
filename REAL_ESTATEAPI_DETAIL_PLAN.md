# RealEstateApi Detail Retrieval Plan

1. **Capture Current Requests**
   - Log the exact JSON payload sent to `/v2/PropertyDetail` for a sample `propertyId`.
   - Try alternative key combinations highlighted in the swagger docs (`id`, `property_id` + `fips`, full address) and record responses.

2. **Validate Against Swagger / Support Samples**
   - Use `https://staging.realestateapi.com/swagger` with our API key to execute the property-detail call for a known `propertyId`.
   - Mirror any working payload in `RealEstateApiListingLoader` once confirmed.

3. **Check MLS Detail Endpoint**
   - Call `/v2/mls/detail` with the `mlsId` returned by the search API to see if DOM lives there instead of property detail.
   - If it does, add a fallback in the loader to pull DOM metrics from the MLS endpoint.

4. **Escalate if Endpoints Still 404**
   - Compile the failing request/response log and contact RealEstateApi support requesting clarification on the required parameters.

5. **Integrate the Working Payload**
   - Update `RealEstateApiListingLoader` to use the confirmed schema, re-run `scripts/probe_realestateapi.py`, and verify DOM fields are populated.
   - Restart the fast-seller training once DOM-to-pending values appear in the REST cache.
